"""Small real end-to-end runs, isolated from thesis result directories."""
import os
import json
from contextlib import nullcontext
from uuid import uuid4
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from joblib import parallel_backend
from utils.experiments.experiment_pipeline import ExperimentPipeline


class PipelineTest(unittest.TestCase):
    def test_classification_and_regression_reports(self):
        previous = Path.cwd()
        directory = Path(__file__).resolve().parent / ('test-output-' + uuid4().hex)
        directory.mkdir()
        with nullcontext(directory):
            try:
                os.chdir(directory)
                rng = np.random.default_rng(72)
                for task, model in [('classification', 'dt'), ('regression', 'linreg')]:
                    frame = pd.DataFrame({'x': rng.normal(size=60), 'x_long': rng.normal(size=60),
                                          'binary': rng.integers(0, 2, 60)})
                    score = 2*frame.x-frame.x_long
                    frame['label'] = (score > 0).astype(int) if task == 'classification' else score
                    frame.to_csv('data.csv', index=False)
                    config = {'dataset_name': task, 'task': task, 'target': 'label',
                              'data_path': 'data.csv', 'split_type': 'random',
                              'xai_methods': ['shap', 'lime', 'permutation'],
                              'xai_sample_size': 4, 'shap_background_size': 5,
                              'shap_permutations': 1, 'lime_num_samples': 100,
                              'stability_seeds': [10, 20], 'masking_seeds': [1,2], 'bootstrap_resamples': 99, 'permutation_resamples':99}
                    with parallel_backend('sequential'):
                        result = ExperimentPipeline(config, [model]).run()
                    methods = result[model]['xai']
                    indices = methods['shap']['evaluation_sample_indices']
                    self.assertEqual(indices, methods['lime']['evaluation_sample_indices'])
                    self.assertEqual(indices, methods['permutation']['evaluation_sample_indices'])
                    self.assertEqual(methods['shap']['output_metadata'], methods['lime']['output_metadata'])
                    self.assertEqual(len(methods['shap']['stability_details']['repeats']), 2)
                    reports = list(Path('results/experiments').glob(f'{task}_*/results.json'))
                    self.assertEqual(len(reports), 1)
                    for method in ['shap', 'lime', 'permutation']:
                        repeat_csv = reports[0].parent / f'{model}_{method}_stability_repeats.csv'
                        self.assertEqual(len(pd.read_csv(repeat_csv)), 2)
                    with reports[0].open(encoding='utf-8') as handle:
                        saved = json.load(handle, parse_constant=lambda v: self.fail(f'Nonstandard JSON: {v}'))
                    self.assertEqual(saved[0]['xai']['shap']['protocol']['version'], 'controlled-xai-v4')
                    self.assertTrue(saved[0]['provenance']['data_sha256'])
                    self.assertTrue(saved[0]['provenance']['code_sha256'])
                    self.assertIn('statistics',saved[0])
                    self.assertIn('algorithm_randomness',saved[0]['statistics'])
                    self.assertTrue((reports[0].parent/f'{model}_randomness_statistics.csv').exists())
                    for method in ['shap','lime','permutation']:
                        details=methods[method]['fidelity']
                        repeat=methods[method]['stability_details']['repeats'][0]
                        self.assertFalse(repeat['randomness_control']['data_changed'])
                        self.assertNotEqual(repeat['algorithm_seed'],repeat['control_algorithm_seed'])
                        self.assertFalse(set(indices)&set(details['sample_indices']))
                        self.assertEqual({r['variant'] for r in details['variants']},{'top','bottom','random'})
                        for error in methods[method]['debug_explanations']:
                            self.assertIn('y_true',error)
                            self.assertIn('prediction',error)
                            if task=='regression':
                                self.assertIn('absolute_error',error)
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main(verbosity=2)

