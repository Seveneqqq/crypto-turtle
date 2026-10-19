"""Fixed rankings, training-derived replacement values, held-out masking controls."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error
from utils.evaluation.feature_mapping import transformed_to_raw


def scalar(value):
    return value.item() if hasattr(value, 'item') else value


def score(y, prediction, task, labels=None):
    if task == 'classification':
        return float(f1_score(y, prediction, labels=labels, average='macro', zero_division=0))
    return float(np.sqrt(mean_squared_error(y, prediction)))


def degradation(baseline, masked, task):
    return baseline-masked if task == 'classification' else masked-baseline


class ControlledMasking:
    def __init__(self, pipeline, training, task):
        self.pipeline, self.task = pipeline, task
        self.labels = pipeline.named_steps['model'].classes_.tolist() if task == 'classification' else None
        used = set(transformed_to_raw(pipeline.named_steps['preprocessing']).values())
        self.reference = {}
        for col in training:
            values = training[col].dropna()
            if col not in used or values.nunique() < 2:
                continue
            numeric = pd.api.types.is_numeric_dtype(training[col])
            if numeric and not set(values.unique()).issubset({0, 1}):
                reference = values.median()
                if np.equal(values, np.floor(values)).all():
                    reference = values.iloc[np.argmin(np.abs(values.to_numpy()-reference))]
            else:
                reference = values.mode().iloc[0]
            self.reference[col] = scalar(reference)

    def mask(self, X, columns):
        changed = X.copy()
        counts = {}
        for col in columns:
            changed[col] = self.reference[col]
            equal = X[col].eq(changed[col]).fillna(False) | (X[col].isna() & changed[col].isna())
            counts[col] = int((~equal).sum())
        return self.pipeline.predict(changed), counts

    def evaluate(self, X, y, importance, k=5, seeds=range(100, 120)):
        if not X.index.equals(y.index) or not X.columns.is_unique:
            raise ValueError('Masking requires aligned indices and unique column names')
        if set(importance) != set(X.columns):
            raise ValueError('Masking rankings must use all raw feature names')
        if len(self.reference) < 2 or k < 1:
            raise ValueError('Need at least two variable used features and positive masking k')
        seeds = list(seeds)
        if len(seeds) < 2 or len(set(seeds)) != len(seeds):
            raise ValueError('Use at least two distinct random-mask seeds')
        effective_k = min(int(k), max(1, len(self.reference)//2))
        ranked = sorted(self.reference, key=lambda c: (-importance[c], c))
        variants = [('top', None, ranked[:effective_k]), ('bottom', None, ranked[-effective_k:])]
        candidates = sorted(self.reference)
        for seed in seeds:
            columns = np.random.default_rng(seed).choice(candidates, effective_k, replace=False).tolist()
            variants.append(('random', int(seed), columns))
        truth = np.asarray(y)
        base_predictions = self.pipeline.predict(X)
        baseline = score(truth, base_predictions, self.task, self.labels)
        records = []
        for name, seed, columns in variants:
            predictions, counts = self.mask(X, columns)
            masked = score(truth, predictions, self.task, self.labels)
            records.append({'variant': name, 'seed': seed, 'selected_columns': columns,
                'modified_columns': [col for col, n in counts.items() if n],
                'changed_rows_per_column': counts, 'score': masked,
                'drop': degradation(baseline, masked, self.task),
                'predictions': np.asarray(predictions).tolist()})
        top, bottom = records[:2]
        random_drop = float(np.mean([r['drop'] for r in records[2:]]))
        return {'baseline_score': baseline, 'perturbed_score': top['score'],
            'ranking_has_signal': bool(np.ptp([importance[c] for c in ranked]) > 0),
            'top_boundary_tie': bool(importance[ranked[effective_k-1]] == importance[ranked[effective_k]]),
            'tie_breaking': 'alphabetical_raw_column_name',
            'fidelity_drop': top['drop'], 'top_minus_random': top['drop']-random_drop,
            'top_minus_bottom': top['drop']-bottom['drop'],
            'requested_k': int(k), 'effective_k': effective_k,
            'eligible_columns': candidates, 'reference_values_training_only': self.reference,
            'sample_indices': X.index.tolist(), 'y_true': truth.tolist(),
            'baseline_predictions': np.asarray(base_predictions).tolist(),
            'task': self.task, 'labels': self.labels, 'variants': records,
            'interpretation': 'Predictive score degradation under replacement; not a complete fidelity measure.'}
