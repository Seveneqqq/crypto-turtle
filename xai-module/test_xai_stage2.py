import unittest
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from utils.preprocessing.pipeline_preprocessor import GenericPreprocessor
from utils.explainers.xai_factory import XaiFactory
from utils.evaluation.paired_evaluation import Perturbation, evaluate_stability, explain_sample, correlation
from utils.evaluation.xai_local import XaiLocal
from utils.evaluation.xai_comparator import XaiComparator


class PairedTests(unittest.TestCase):
    def data(self, task='classification', multiclass=False, tree=False):
        rng = np.random.default_rng(7)
        X = pd.DataFrame({'volume': rng.normal(size=80), 'quote_volume': rng.normal(size=80),
                          'binary': rng.integers(0, 2, 80), 'region': ['A_B', 'C']*40})
        score = 3*X.volume - 2*X.quote_volume
        y = pd.Series(np.digitize(score, [-1, 1]) if multiclass else (score > 0).astype(int))
        model = RandomForestClassifier(n_estimators=4, max_depth=3, random_state=1) if tree else LogisticRegression()
        if task == 'regression':
            y, model = score, LinearRegression()
        pre = GenericPreprocessor('y')._build_transformer(X.iloc[:60])
        pipe = Pipeline([('preprocessing', pre), ('model', model)]).fit(X.iloc[:60], y.iloc[:60])
        return X, y, pipe

    def config(self):
        return {'shap_background_size': 8, 'shap_permutations': 1, 'lime_num_samples': 200}

    def test_fixed_output_and_shap_additivity(self):
        for task, multi, tree in [('classification', False, False), ('classification', True, False),
                                  ('classification', True, True), ('regression', False, False)]:
            X, y, pipe = self.data(task, multi, tree)
            cfg = self.config()
            cfg['xai_class_index'] = 0
            shap = XaiFactory.create('shap', pipe, X.iloc[:60], task, cfg)
            lime = XaiFactory.create('lime', pipe, X.iloc[:60], task, cfg)
            self.assertEqual(shap.output_metadata(), lime.output_metadata())
            background = shap.background.copy()
            sample = X.iloc[60:62]
            first = shap.explain(sample, random_state=9)
            np.testing.assert_allclose(first.sum(axis=1)+shap.last_base_values,
                                       shap.predict_output(shap.transform(sample)), atol=1e-6)
            np.testing.assert_allclose(first, shap.explain(sample, random_state=9))
            weights = lime.explain(sample.iloc[0], random_state=9)
            self.assertEqual(weights, lime.explain(sample.iloc[0], random_state=9))
            self.assertEqual(set(weights), set(lime.feature_names))
            np.testing.assert_array_equal(background, shap.background)

    def test_signed_lime_and_class_metadata(self):
        X, y, pipe = self.data('regression')
        lime = XaiFactory.create('lime', pipe, X.iloc[:60], 'regression', self.config())
        weights = lime.explain(X.iloc[60])
        self.assertTrue(any(v < 0 for v in weights.values()))
        self.assertTrue(any(v > 0 for v in weights.values()))
        local = XaiLocal.generate_local_explanations('lime', lime, X.iloc[60:61])
        self.assertEqual(local[0]['output_metadata']['output'], 'prediction')
        self.assertTrue(any(v['value'] < 0 for v in local[0]['top_features']))

    def test_noise_reproducible_scaled_and_valid(self):
        X, y, pipe = self.data()
        X['constant'] = 3
        X['all_missing'] = np.nan
        X['ordinal'] = np.tile([1, 5, 20, 50], 20)
        noise = Perturbation(X.iloc[:60], .05, .3)
        a, changes = noise.apply(X.iloc[60:], 23)
        b, _ = noise.apply(X.iloc[60:], 23)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(set(a.binary.unique()), {0, 1})
        self.assertTrue(set(a.region).issubset(set(X.region)))
        self.assertTrue((a.constant == 3).all())
        self.assertTrue(a.all_missing.isna().all())
        levels = [1, 5, 20, 50]
        for old, new in zip(X.iloc[60:].ordinal, a.ordinal):
            self.assertLessEqual(abs(levels.index(old)-levels.index(new)), 1)
        self.assertEqual(list(a.index), list(X.iloc[60:].index))
        scaled = X.copy()
        scaled.volume *= 1000
        c, _ = Perturbation(scaled.iloc[:60], .05, .3).apply(scaled.iloc[60:], 23)
        np.testing.assert_allclose(c.volume, a.volume*1000)

    def test_paired_zero_noise_and_common_feature_space(self):
        X, y, pipe = self.data()
        samples = X.iloc[60:63]
        all_names = []
        for method in ['shap', 'lime', 'permutation']:
            explainer = XaiFactory.create(method, pipe, X.iloc[:60], 'classification', self.config())
            with parallel_backend('sequential'):
                imp, _ = explain_sample(method, explainer, samples, y.loc[samples.index], 42)
                report = evaluate_stability(method, explainer, samples, y.loc[samples.index],
                             Perturbation(X.iloc[:60], 0, 0), [10, 20])
            all_names.append(set(imp))
            for repeat in report['repeats']:
                self.assertEqual(repeat['baseline_importance'], repeat['perturbed_importance'])
                self.assertIn(repeat['global_spearman'], [1., None])
                self.assertFalse(any(repeat['changed_rows_per_column'].values()))
            self.assertEqual(report['sample_indices'], samples.index.tolist())
        self.assertTrue(all(names == set(X.columns) for names in all_names))

    def test_degenerate_correlation_and_overlap(self):
        self.assertTrue(np.isnan(correlation({'a':0, 'b':0}, {'a':0, 'b':0})))
        self.assertEqual(XaiComparator.top_k_overlap({'a':2, 'b':1}, {'a':2, 'b':1}), 1)
        with self.assertRaises(ValueError):
            correlation({'num__a':1}, {'a':1})


if __name__ == '__main__':
    unittest.main(verbosity=2)
