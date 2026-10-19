import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from utils.evaluation.paired_evaluation import evaluate_stability, Perturbation


class RandomnessSeparationTest(unittest.TestCase):
    def run_case(self, use_seed, use_data):
        X=pd.DataFrame({'a':[1.1,2.2,3.3], 'b':[4.4,5.5,6.6], 'c':[7.7,8.8,9.9]},index=[7,4,9])
        y=pd.Series([0,1,0],index=X.index)
        model_marker=object()
        calls=[]
        def explain(method, explainer, data, target, seed):
            self.assertIs(explainer,model_marker)
            self.assertTrue(data.index.equals(X.index))
            self.assertTrue(target.index.equals(X.index))
            calls.append((data.copy(),seed))
            values=np.array([1.,2.,3.])
            if use_seed:
                values += np.random.default_rng(seed).uniform(0,3,3)
            if use_data:
                values += data.mean().to_numpy()
            return dict(zip(X.columns,values)),pd.DataFrame(np.tile(values,(len(X),1)),index=X.index,columns=X.columns)
        with patch('utils.evaluation.paired_evaluation.explain_sample',side_effect=explain):
            result=evaluate_stability('shap',model_marker,X,y,Perturbation(X,.2,0),[10,20])
        for i in range(2):
            baseline,control,noisy=calls[3*i:3*i+3]
            pd.testing.assert_frame_equal(baseline[0],control[0])
            self.assertEqual(baseline[1],noisy[1])
            self.assertNotEqual(baseline[1],control[1])
            self.assertFalse(baseline[0].equals(noisy[0]))
        return result

    def test_algorithm_only_variation_not_misattributed_to_data(self):
        result=self.run_case(True,False)
        self.assertEqual(result['data_relative_l1_summary']['mean'],0)
        self.assertGreater(result['randomness_relative_l1_summary']['mean'],0)

    def test_data_only_variation_not_misattributed_to_algorithm(self):
        result=self.run_case(False,True)
        self.assertGreater(result['data_relative_l1_summary']['mean'],0)
        self.assertEqual(result['randomness_relative_l1_summary']['mean'],0)

    def test_seed_schedule_and_measurements_reproducible(self):
        first=self.run_case(True,True)
        second=self.run_case(True,True)
        self.assertEqual(first,second)


if __name__=='__main__':
    unittest.main(verbosity=2)
