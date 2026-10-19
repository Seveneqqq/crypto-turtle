import copy
import json
from pathlib import Path
from contextlib import nullcontext
from uuid import uuid4
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from utils.evaluation.output_audit_v5 import ScalarOutput, OutputPermutation, masking_reports, error_explanations
from utils.explainers.xai_factory import XaiFactory
from utils.evaluation.statistical_analysis import masking_inference
import repair_experiments as runner


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(22)
        cls.X=pd.DataFrame(rng.normal(size=(90,4)),columns=list('abcd'))
        cls.y=pd.Series(np.argmax(np.column_stack([cls.X.a,cls.X.b,-cls.X.a-cls.X.b]),axis=1))
        cls.pipe=Pipeline([('preprocessing',ColumnTransformer([('num',StandardScaler(),list('abcd'))])),
                           ('model',LogisticRegression(random_state=42))]).fit(cls.X,cls.y)

    def test_scalar_target_is_selected_class_not_argmax(self):
        out=ScalarOutput(self.pipe,'classification',1)
        np.testing.assert_allclose(out.predict(self.X),self.pipe.predict_proba(self.X)[:,1])
        np.testing.assert_array_equal(out.target(self.y),self.y==1)

    def test_masking_root_brier_and_output_change_are_distinct(self):
        out=ScalarOutput(self.pipe,'classification',1)
        X=self.X.iloc[:40];y=self.y.iloc[:40]
        perf,sens=masking_reports(out,self.X,X,y,dict(a=4.,b=3.,c=2.,d=1.),k=1,seeds=[1,2])
        base=out.predict(X)
        self.assertAlmostEqual(perf['baseline_score'],np.sqrt(np.mean(((y==1)-base)**2)))
        self.assertEqual(sens['baseline_score'],0.)
        self.assertEqual(sens['y_true'],base.tolist())
        for v in sens['variants']:
            self.assertAlmostEqual(v['drop'],np.sqrt(np.mean((np.array(v['predictions'])-base)**2)))
        stats=masking_inference({'shap':{'fidelity':sens},'lime':{'fidelity':copy.deepcopy(sens)}},resamples=39)
        self.assertEqual(stats['comparisons'][-1]['effect'],0.)

    def test_pfi_responds_to_signal_and_names_align(self):
        out=ScalarOutput(self.pipe,'classification',0)
        result=OutputPermutation(out).explain(self.X,self.y)
        self.assertEqual(set(result),set(self.X))
        self.assertGreater(result['a'],result['d'])

    def test_error_explains_predicted_class_and_restores_target(self):
        X=self.X.iloc[:3];pred=self.pipe.predict(X)
        wrong=pd.Series((pred+1)%3,index=X.index)
        for method in ['shap','lime']:
            ex=XaiFactory.create(method,self.pipe,self.X,config={'xai_class_index':2,'shap_background_size':5,'shap_permutations':1,'lime_num_samples':150})
            records=error_explanations(method,ex,X,wrong,pred,'classification')
            self.assertEqual(ex.class_index,2)
            self.assertEqual(len(records),3)
            for r in records:
                self.assertEqual(r['output_metadata']['class_label'],r['prediction'])
                self.assertNotEqual(r['y_true'],r['prediction'])
                if method=='lime':self.assertIn('local_surrogate_fit_score',r)

    def test_regression_masking_uses_shared_reference(self):
        pipe=Pipeline([('preprocessing',ColumnTransformer([('num',StandardScaler(),list('abcd'))])),('model',LinearRegression())]).fit(self.X,self.X.a*2)
        out=ScalarOutput(pipe,'regression')
        a,b=masking_reports(out,self.X,self.X,self.X.a*2,dict(a=4.,b=3.,c=2.,d=1.),k=1,seeds=[1,2])
        self.assertLess(a['baseline_score'],1e-12)
        self.assertGreater(b['fidelity_drop'],1.)

    def test_checkpoint_resume_and_reuse_without_old_repetitions(self):
        config={'dataset_name':'tiny','task':'classification','xai_sample_size':4,
            'masking_sample_size':30,'shap_background_size':5,'shap_permutations':1,
            'lime_num_samples':150,'stability_seeds':[101,102],'masking_seeds':[501,502],
            'bootstrap_resamples':39,'split_type':'random'}
        fitted={'pipeline':self.pipe,'X_train':self.X,'X_test':self.X,'y_test':self.y,
            'y_pred':self.pipe.predict(self.X),'metrics':{'accuracy':.9},'data_preparation':{'chronology':'random_split'}}
        scratch=Path(__file__).parent/'test_artifacts'/uuid4().hex
        scratch.mkdir(parents=True)
        with nullcontext(scratch) as tmp:
            folder=Path(tmp)/'new'
            with patch.object(runner.ExperimentRunner,'run',return_value=fitted):
                runner.run_job('tiny','logreg',config,None,None,folder,{'fingerprint':'test'})
            first=runner.read(folder/'results.json')
            with patch.object(runner.ExperimentRunner,'run',side_effect=AssertionError('must not retrain')):
                runner.run_job('tiny','logreg',config,None,None,folder,{'fingerprint':'test'})
            state=runner.read(folder/'checkpoint.json')
            state['files'].pop('results.json');state['files'].pop('shap.json')
            runner.save(folder/'checkpoint.json',state)
            with patch.object(runner.ExperimentRunner,'run',side_effect=AssertionError('must not retrain')),patch.object(runner,'evaluate_stability',side_effect=AssertionError('must reuse completed stability')):
                runner.run_job('tiny','logreg',config,None,None,folder,{'fingerprint':'test'})
            old={'model_metrics':fitted['metrics'],'data_preparation':fitted['data_preparation'],'task':'classification','xai':copy.deepcopy(first['xai'])}
            mask_indices=old['xai']['shap']['performance_masking']['sample_indices']
            old['xai']['shap']['fidelity']={'sample_indices':mask_indices,'baseline_predictions':self.pipe.predict(self.X.loc[mask_indices]).tolist()}
            original=runner.evaluate_stability
            def checked(method,*args,**kwargs):
                if method!='permutation': raise AssertionError('SHAP/LIME repeats must be reused')
                return original(method,*args,**kwargs)
            with patch.object(runner.ExperimentRunner,'run',return_value=fitted),patch.object(runner,'evaluate_stability',side_effect=checked):
                runner.run_job('tiny','logreg',config,old,{'sha256':'synthetic'},Path(tmp)/'reused',{'fingerprint':'test'})
            result=runner.read(Path(tmp)/'reused/results.json')
            self.assertEqual(result['xai']['shap']['stability_origin']['kind'],'reused_after_reproduction_checks')
            old['xai']['shap']['fidelity']['baseline_predictions'][0]=999
            with self.assertRaises(ValueError):runner.validate_model(old,fitted)

    def test_changed_sources_rejected(self):
        original={'config':{},'data_sha256':'a','packages':{},'python':'3','code_sha256':dict.fromkeys(runner.REUSE_FILES,'x')}
        current=copy.deepcopy(original);current['data_sha256']='b'
        with self.assertRaises(ValueError):runner.validate_source({'provenance':original},current,{})

    def test_nan_model_parameter_matches_serialized_null(self):
        self.assertEqual(runner.clean({'missing':float('nan')}),{'missing':None})


if __name__=='__main__': unittest.main(verbosity=2)
