import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from joblib import parallel_backend
from utils.preprocessing.pipeline_preprocessor import GenericPreprocessor
from utils.explainers.permutation_explainer import PermutationExplainer
from utils.evaluation.xai_fidelity import XaiFidelity
from utils.evaluation.feature_mapping import transformed_to_raw


class ProcedureTests(unittest.TestCase):
    def test_training_median_and_scaling_disabled(self):
        df = pd.DataFrame({'x': [2., 4., np.nan, 6., 1000., np.nan],
                           'y': [0, 1, 0, 1, 0, 1]})
        prep = GenericPreprocessor('y', split_type='time', test_size=1/3, scaling=False)
        train, test, yt, yv, transform = prep.prepare(df)
        transform.fit(train)
        self.assertEqual(transform.named_transformers_['num']['imputation'].statistics_[0], 4.)
        np.testing.assert_array_equal(transform.transform(test).ravel(), [1000., 4.])

    def test_finance_tail_and_gap(self):
        for horizon in [1, 5]:
            for target_type, target in [('direction', 'target_direction'), ('return', 'target'), ('future_close', 'target')]:
                df = pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=30, freq='min'),
                                   'close': np.arange(1., 31.)})
                prep = GenericPreprocessor(target, split_type='time', finance_horizon=horizon,
                                           finance_target_type=target_type)
                train, test, yt, yv, _ = prep.prepare(df)
                self.assertEqual(test.index[-1], 29-horizon)
                self.assertLess(train.index[-1]+horizon, test.index[0])
                self.assertEqual(prep.split_metadata_['gap_rows'], horizon)
                if target_type == 'direction':
                    self.assertTrue((yt == 1).all() and (yv == 1).all())

    def test_missing_future_price_and_medical_label(self):
        df = pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=20),
                           'close': np.arange(1., 21.)})
        df.loc[10, 'close'] = np.nan
        prep = GenericPreprocessor('target_direction', split_type='time')
        engineered = prep._feature_engineering(prep._clean(df))
        self.assertNotIn(9, engineered.index)
        self.assertNotIn(10, engineered.index)
        self.assertNotIn(19, engineered.index)
        medical = GenericPreprocessor('Diabetes_012')
        clean = medical._clean(pd.DataFrame({'Diabetes_012': [0, np.nan, 2]}))
        self.assertEqual(clean['Diabetes_012'].tolist(), [0, 1])

    def make_pipeline(self):
        X = pd.DataFrame({'volume': [1., 2., 3., 4., 5., 6.],
                          'district': ['A_1', 'B', 'A_1', 'B', 'A_1', 'B'],
                          'quote_volume': [10., 20., 30., 40., 50., 60.],
                          'flag': [0, 1, 0, 1, 0, 1]})
        y = pd.Series([0, 1, 0, 1, 0, 1])
        transform = GenericPreprocessor('y')._build_transformer(X)
        pipeline = Pipeline([('preprocessing', transform), ('model', DecisionTreeClassifier(random_state=42))])
        pipeline.fit(X, y)
        return X, y, pipeline

    def test_pfi_raw_names(self):
        X, y, pipeline = self.make_pipeline()
        explainer = PermutationExplainer(pipeline)
        with parallel_backend('sequential'):
            result = explainer.explain(X, y)
        self.assertEqual(result.feature_names, list(X.columns))
        ranked = explainer.get_global_importance(result)
        for i, name in enumerate(X.columns):
            self.assertEqual(ranked[name], result.importances_mean[i])

    def test_exact_mapping_and_recorded_changes(self):
        X, y, pipeline = self.make_pipeline()
        mapping = transformed_to_raw(pipeline['preprocessing'])
        self.assertEqual(mapping['cat__district_A_1'], 'district')
        result = XaiFidelity.feature_removal_test(pipeline, X, y,
                    {'num__volume': 1}, k=1)
        self.assertEqual(result['modified_columns'], ['volume'])
        self.assertEqual(result['changed_rows_per_column'], {'volume': 6})
        result = XaiFidelity.feature_removal_test(pipeline, X, y,
                    {'quote_volume': 1}, k=1, feature_space='raw')
        self.assertEqual(result['modified_columns'], ['quote_volume'])
        with self.assertRaises(ValueError):
            XaiFidelity.feature_removal_test(pipeline, X, y, {'vol': 1}, k=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)

