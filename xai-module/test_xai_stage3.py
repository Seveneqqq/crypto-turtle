import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from utils.preprocessing.pipeline_preprocessor import GenericPreprocessor
from utils.evaluation.controlled_masking import ControlledMasking, score
from utils.evaluation.statistical_analysis import bootstrap_indices, bootstrap_scores, holm, masking_inference, noise_comparisons
from utils.evaluation.xai_debugger import XaiDebugger


class FinalProcedureTests(unittest.TestCase):
    def setup_mask(self):
        rng=np.random.default_rng(15)
        X=pd.DataFrame({'signal':rng.normal(size=140),'nuisance':rng.normal(size=140),
                        'constant':np.ones(140)})
        y=4*X.signal
        transform=GenericPreprocessor('label')._build_transformer(X.iloc[:80])
        pipeline=Pipeline([('preprocessing',transform),('model',LinearRegression())]).fit(X.iloc[:80],y.iloc[:80])
        masker=ControlledMasking(pipeline,X.iloc[:80],'regression')
        report=masker.evaluate(X.iloc[80:],y.iloc[80:],{'signal':4.,'nuisance':0.,'constant':0.},k=5,seeds=[1,2,3,4])
        return X,y,masker,report

    def test_masking_controls_reference_and_cardinality(self):
        X,y,masker,report=self.setup_mask()
        self.assertEqual(report['effective_k'],1)
        self.assertNotIn('constant',report['eligible_columns'])
        self.assertEqual(report['reference_values_training_only']['signal'],X.iloc[:80].signal.median())
        self.assertGreater(report['top_minus_bottom'],1)
        self.assertTrue(all(len(r['selected_columns'])==1 for r in report['variants']))
        second=masker.evaluate(X.iloc[80:],y.iloc[80:],{'signal':0.,'nuisance':4.,'constant':0.},k=5,seeds=[1,2,3,4])
        self.assertEqual([r['selected_columns'] for r in report['variants'][2:]],
                         [r['selected_columns'] for r in second['variants'][2:]])

    def test_bootstrap_metric_matches_original(self):
        truth=np.array([0,0,1,1,2,2])
        pred=np.array([0,1,1,0,1,2])
        indices=np.array([[0,1,2,3,4,5],[0,0,1,2,2,3]])
        for task in ['classification','regression']:
            boot=bootstrap_scores(truth,pred,task,[0,1,2],indices)
            expected=[score(truth[i],pred[i],task,[0,1,2]) for i in indices]
            np.testing.assert_allclose(boot,expected)

    def test_time_blocks_and_reproducibility(self):
        samples=bootstrap_indices(40,30,17,5)
        np.testing.assert_array_equal(samples,bootstrap_indices(40,30,17,5))
        for row in samples:
            for block in row.reshape(-1,5):
                np.testing.assert_array_equal(np.diff(block),np.ones(4))

    def test_holm_known_values(self):
        rows=holm([{'p_value':.01},{'p_value':.04},{'p_value':.03},{'p_value':None}],'family')
        np.testing.assert_allclose([r['p_holm'] for r in rows[:3]],[.03,.06,.06])
        self.assertIsNone(rows[3]['p_holm'])

    def test_identical_methods_have_no_superiority(self):
        X,y,masker,report=self.setup_mask()
        methods={'a':{'fidelity':report},'b':{'fidelity':report}}
        result=masking_inference(methods,temporal=True,horizon=2,resamples=99,seed=1)
        between=[r for r in result['comparisons'] if r['metric']=='difference_in_top_minus_random'][0]
        self.assertEqual(between['effect'],0)
        self.assertEqual(between['p_value'],1)
        broken=dict(report)
        broken['sample_indices']=list(reversed(report['sample_indices']))
        with self.assertRaises(ValueError):
            masking_inference({'a':{'fidelity':report},'b':{'fidelity':broken}},resamples=99)

    def test_noise_inference_and_undefined_pairs(self):
        records=[{'seed':i,'global_spearman':.8+i/1000} for i in range(12)]
        methods={m:{'stability_details':{'repeats':records}} for m in ['shap','lime']}
        result=noise_comparisons(methods,resamples=99)[0]
        self.assertEqual(result['p_value'],1)
        self.assertEqual(result['ci95'],[0,0])
        methods['lime']={'stability_details':{'repeats':[{'seed':i,'global_spearman':None} for i in range(12)]}}
        self.assertEqual(noise_comparisons(methods,resamples=99)[0]['status'],'insufficient_valid_repeats')
        methods['lime']={'stability_details':{'repeats':[{'seed':i,'global_spearman':.5+i/2000} for i in range(12)]}}
        alternative=noise_comparisons(methods,resamples=499)[0]
        self.assertGreater(alternative['effect'],0)
        self.assertLess(alternative['p_value'],.05)

    def test_error_reports_contain_actual_prediction_and_error(self):
        X=pd.DataFrame({'x':[1.,2.,3.]},index=[10,20,30])
        y=pd.Series([1.,2.,3.],index=X.index)
        records=XaiDebugger.explain_errors('regression','permutation',None,X,y,np.array([1.,2.,9.]))
        self.assertEqual(records[0]['sample_index'],30)
        self.assertEqual(records[0]['y_true'],3)
        self.assertEqual(records[0]['prediction'],9)
        self.assertEqual(records[0]['absolute_error'],6)

    def test_chronology_and_finance_milliseconds(self):
        crime=pd.DataFrame({'OCCURRED_ON_DATE':pd.date_range('2020-01-01',periods=10)[::-1],
                            'OFFENSE_CODE_GROUP':['A','B']*5,'x':np.arange(10.)})
        prep=GenericPreprocessor('OFFENSE_CODE_GROUP',split_type='time')
        train,test,_,_,_=prep.prepare(crime)
        self.assertEqual(train.index.tolist(),list(range(9,1,-1)))
        self.assertEqual(test.index.tolist(),[1,0])
        finance=pd.DataFrame({'timestamp':pd.date_range('2024-01-01',periods=10),
                              'close':np.arange(1.,11.),'close_time':[1704067200000]*10})
        transformed=GenericPreprocessor('target',split_type='time')._feature_engineering(finance)
        self.assertTrue((transformed.close_time_year==2024).all())


if __name__=='__main__':
    unittest.main(verbosity=2)
