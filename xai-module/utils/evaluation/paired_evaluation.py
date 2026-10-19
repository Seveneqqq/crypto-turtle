"""Paired stability on a fixed held-out sample; no model retraining."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from utils.evaluation.feature_mapping import transformed_to_raw


def correlation(first, second):
    if set(first) != set(second):
        raise ValueError('Ranking feature sets must be identical')
    names = sorted(first)
    a, b = [first[k] for k in names], [second[k] for k in names]
    if len(names) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float('nan')
    return float(spearmanr(a, b).statistic)


def summary(values):
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return {'mean': float(finite.mean()) if len(finite) else None,
            'std': float(finite.std(ddof=1)) if len(finite) > 1 else None,
            'valid_repeats': int(len(finite)), 'total_repeats': len(values)}


class Perturbation:
    def __init__(self, training, level=.01, discrete_probability=.01, feature_types=None):
        if level < 0 or not 0 <= discrete_probability <= 1:
            raise ValueError('Invalid perturbation magnitude/probability')
        self.level, self.probability = level, discrete_probability
        self.spec = {}
        overrides = feature_types or {}
        if set(overrides) - set(training.columns):
            raise ValueError('Unknown feature in stability_feature_types')
        for col in training:
            observed = training[col].dropna()
            numeric = pd.api.types.is_numeric_dtype(training[col])
            values = np.sort(observed.unique()) if numeric else observed.unique()
            if numeric and len(values) and not np.isfinite(values).all():
                raise ValueError(f'Non-finite training feature: {col}')
            integer = numeric and len(values) and np.all(np.asarray(values) == np.floor(values))
            inferred = 'ordinal' if integer and len(values) > 2 else ('discrete' if integer or not numeric else 'continuous')
            kind = overrides.get(col, inferred)
            if kind not in ('continuous', 'discrete', 'ordinal', 'fixed') or (kind in ('continuous', 'ordinal') and not numeric):
                raise ValueError(f'Invalid stability type for {col}')
            self.spec[col] = {'kind': kind, 'values': values,
                              'scale': float(observed.std(ddof=0)) if numeric and len(values) else 0.}

    def apply(self, X, seed):
        rng = np.random.default_rng(seed)
        noisy = X.copy()
        changed = {}
        for col, spec in self.spec.items():
            vals = spec['values']
            if len(vals) < 2 or spec['kind'] == 'fixed':
                continue
            present = X[col].notna().to_numpy()
            if spec['kind'] == 'continuous':
                original = X[col].to_numpy(dtype=float)
                result = original + rng.normal(0, self.level*spec['scale'], len(X))
                result = np.clip(result, np.minimum(original, vals.min()), np.maximum(original, vals.max()))
                noisy[col] = result
            else:
                selected = present & (rng.random(len(X)) < self.probability)
                for position in np.flatnonzero(selected):
                    original = X.iloc[position][col]
                    alternatives = vals[vals != original]
                    if spec['kind'] == 'ordinal':
                        lower, upper = vals[vals < original], vals[vals > original]
                        alternatives = np.asarray(([lower[-1]] if len(lower) else []) +
                                                  ([upper[0]] if len(upper) else []))
                    noisy.iloc[position, noisy.columns.get_loc(col)] = rng.choice(alternatives)
            equal = X[col].eq(noisy[col]).fillna(False) | (X[col].isna() & noisy[col].isna())
            changed[col] = int((~equal).sum())
        return noisy, changed

    def metadata(self):
        return {'continuous_noise_fraction_of_training_std': self.level,
                'discrete_replacement_probability': self.probability,
                'features': {col: {'kind': s['kind'], 'training_std': s['scale'],
                                   'observed_levels': len(s['values'])} for col, s in self.spec.items()}}


def explain_sample(method, explainer, X, y, seed):
    if method == 'permutation':
        result = explainer.explain(X, y, random_state=seed)
        return explainer.get_global_importance(result), None
    if method == 'shap':
        matrix = explainer.explain(X, random_state=seed)
    else:
        rows = [explainer.explain(X.iloc[i], random_state=(seed+i) % (2**32)) for i in range(len(X))]
        matrix = np.asarray([[row.get(name, 0.) for name in explainer.feature_names] for row in rows])
    provenance = transformed_to_raw(explainer.preprocessor)
    signed = pd.DataFrame(matrix, index=X.index, columns=explainer.feature_names)
    magnitudes = pd.DataFrame(0., index=X.index, columns=X.columns)
    for name in signed.columns:
        magnitudes[provenance[name]] += signed[name].abs()
    importance = magnitudes.mean().to_dict()
    return dict(sorted(importance.items(), key=lambda pair: pair[1], reverse=True)), magnitudes


def relative_l1(first, second):
    names = sorted(first)
    a = np.asarray([first[k] for k in names])
    b = np.asarray([second[k] for k in names])
    denominator = np.abs(a).sum()+np.abs(b).sum()
    return float(np.abs(a-b).sum()/denominator) if denominator else 0.


def compare_explanations(first, second, local_first, local_second):
    rho = correlation(first,second)
    local_rho, local_distance = [], []
    if local_first is not None:
        if not local_first.index.equals(local_second.index):
            raise ValueError('Local explanation observation IDs differ')
        for i in range(len(local_first)):
            a,b = local_first.iloc[i].to_dict(),local_second.iloc[i].to_dict()
            value = correlation(a,b)
            local_rho.append(value if np.isfinite(value) else None)
            local_distance.append(relative_l1(a,b))
    return {'global_spearman':rho if np.isfinite(rho) else None,
            'global_relative_l1':relative_l1(first,second),
            'local_spearman':local_rho, 'local_relative_l1':local_distance,
            'local_summary':summary([v if v is not None else float('nan') for v in local_rho])}


def evaluate_stability(method, explainer, X, y, perturbation, seeds):
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('Use at least two distinct stability_seeds')
    records = []
    for seed in seeds:
        generated = np.random.SeedSequence([int(seed), 5741]).generate_state(2)
        algorithm_seed, control_seed = map(int, generated)
        print(f'  Stability + randomness control {method}: seed {seed}', flush=True)
        baseline, local_baseline = explain_sample(method, explainer, X, y, algorithm_seed)
        control, local_control = explain_sample(method, explainer, X, y, control_seed)
        noisy, changes = perturbation.apply(X, seed)
        perturbed, local_perturbed = explain_sample(method, explainer, noisy, y, algorithm_seed)
        data_comparison = compare_explanations(baseline,perturbed,local_baseline,local_perturbed)
        control_comparison = compare_explanations(baseline,control,local_baseline,local_control)
        records.append({'seed':int(seed), 'algorithm_seed':algorithm_seed,
                        'control_algorithm_seed':control_seed, **data_comparison,
                        'baseline_importance':baseline, 'perturbed_importance':perturbed,
                        'changed_rows_per_column':changes,
                        'randomness_control':{**control_comparison,
                            'data_changed':False, 'importance':control}})
    def metric_summary(key, control=False):
        values=[(r['randomness_control'] if control else r)[key] for r in records]
        return summary([v if v is not None else float('nan') for v in values])
    return {'global_summary':metric_summary('global_spearman'), 'repeats':records,
            'data_relative_l1_summary':metric_summary('global_relative_l1'),
            'randomness_summary':metric_summary('global_spearman',True),
            'randomness_relative_l1_summary':metric_summary('global_relative_l1',True),
            'sample_indices':X.index.tolist(), 'perturbation':perturbation.metadata(),
            'scope':'fixed_model_sample_background; paired_data_change_and_separate_algorithm_seed_control',
            'note':'Data arm holds algorithm seed fixed; control arm holds data fixed. '
                   'These are separate sensitivity measurements, not an additive variance decomposition. '
                   'Scores describe global/local magnitude rankings, not all properties of signed explanations.'}
