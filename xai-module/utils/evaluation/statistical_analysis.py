"""Conditional paired comparisons; explicitly separate noise and observation units."""
import itertools
import numpy as np
from scipy.stats import permutation_test


def holm(rows, family):
    valid = sorted([(i, r['p_value']) for i, r in enumerate(rows) if r.get('p_value') is not None], key=lambda x:x[1])
    previous = 0.
    for rank, (i, p) in enumerate(valid):
        previous = max(previous, min(1., p*(len(valid)-rank)))
        rows[i]['p_holm'] = previous
    for row in rows:
        row['family'] = family
        row.setdefault('p_holm', None)
    return rows


def noise_comparisons(method_results, seed=42, resamples=4999):
    rows = []
    for first, second in itertools.combinations(method_results, 2):
        a = {r['seed']:r['global_spearman'] for r in method_results[first]['stability_details']['repeats']}
        b = {r['seed']:r['global_spearman'] for r in method_results[second]['stability_details']['repeats']}
        if set(a) != set(b):
            raise ValueError('Unpaired stability seed sets')
        seeds = sorted(k for k in a if a[k] is not None and b[k] is not None)
        differences = np.array([a[k]-b[k] for k in seeds])
        row = {'metric':'stability_spearman', 'first':first, 'second':second,
               'n_pairs':len(seeds), 'paired_seeds':seeds, 'unit':'independent_perturbation_seed',
               'scope':'conditional_on_fixed_model_sample_background_and_noise_protocol', 'p_value':None,
               'effect':float(differences.mean()) if len(seeds) else None,
               'ci95':None, 'test':'paired_sign_swap_mean_difference'}
        if len(seeds) < 10:
            row['status'] = 'insufficient_valid_repeats'
        elif np.all(differences == 0):
            row.update(status='identical', p_value=1., ci95=[0.,0.])
        else:
            rng = np.random.default_rng(seed)
            means = differences[rng.integers(0,len(seeds),size=(resamples,len(seeds)))].mean(axis=1)
            row['ci95'] = np.quantile(means,[.025,.975]).tolist()
            test = permutation_test((differences,), np.mean, permutation_type='samples',
                                    alternative='two-sided', n_resamples=resamples,
                                    random_state=seed, vectorized=False)
            row.update(status='ok_under_pair_exchangeability', p_value=float(test.pvalue))
        rows.append(row)
    return holm(rows, 'stability_pairs_within_dataset_model')


def bootstrap_indices(n, draws, seed, block_length=1):
    if block_length < 1 or block_length > n:
        raise ValueError('Invalid bootstrap block length')
    rng = np.random.default_rng(seed)
    if block_length == 1:
        return rng.integers(0,n,size=(draws,n))
    starts = rng.integers(0,n-block_length+1,size=(draws,int(np.ceil(n/block_length))))
    return (starts[:,:,None]+np.arange(block_length)).reshape(draws,-1)[:,:n]


def bootstrap_scores(truth, predictions, task, labels, indices):
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    if task == 'regression':
        errors = (truth-predictions)**2
        return np.sqrt(errors[indices].mean(axis=1))
    lookup = {label:i for i,label in enumerate(labels)}
    y = np.array([lookup[v] for v in truth])
    p = np.array([lookup[v] for v in predictions])
    width = len(labels)
    combined = y*width+p
    scores = []
    for sample in indices:
        matrix = np.bincount(combined[sample], minlength=width*width).reshape(width,width)
        denominator = matrix.sum(axis=0)+matrix.sum(axis=1)
        scores.append(np.divide(2*np.diag(matrix),denominator,
                                out=np.zeros(width,dtype=float),where=denominator!=0).mean())
    return np.asarray(scores)


def masking_inference(method_results, temporal=False, horizon=1, resamples=1999, seed=42, block_length=None):
    first = next(iter(method_results.values()))['fidelity']
    n = len(first['y_true'])
    length = int(block_length or max(int(horizon), round(n**(1/3)))) if temporal else 1
    length = min(length, n)
    indices = bootstrap_indices(n, resamples, seed, length)
    task = first['task']
    baseline = bootstrap_scores(first['y_true'],first['baseline_predictions'],task,first['labels'],indices)
    distributions, points, intervals = {}, {}, []
    for method, result in method_results.items():
        report = result['fidelity']
        if (report['sample_indices'] != first['sample_indices'] or report['y_true'] != first['y_true']
                or report['baseline_predictions'] != first['baseline_predictions'] or report['labels'] != first['labels']):
            raise ValueError('Masking inference requires paired observations and the same model')
        drops = []
        for record in report['variants']:
            masked = bootstrap_scores(report['y_true'],record['predictions'],task,report['labels'],indices)
            drops.append(baseline-masked if task=='classification' else masked-baseline)
        random = np.asarray(drops[2:])
        control_draws = np.random.default_rng(seed+1).integers(0,len(random),size=(resamples,len(random)))
        random_mean = random[control_draws, np.arange(resamples)[:,None]].mean(axis=1)
        distributions[method] = {'top':drops[0], 'top_minus_bottom':drops[0]-drops[1],
                                 'top_minus_random':drops[0]-random_mean}
        points[method] = {'top':report['fidelity_drop'], 'top_minus_bottom':report['top_minus_bottom'],
                          'top_minus_random':report['top_minus_random']}
        for metric, values in distributions[method].items():
            intervals.append({'method':method, 'metric':metric, 'estimate':points[method][metric],
                'ci95':np.quantile(values,[.025,.975]).tolist(), 'bootstrap_std':float(values.std(ddof=1))})
    rows = []
    def append(name, a, b, observed, distribution):
        row = {'metric':name, 'first':a, 'second':b, 'effect':float(observed),
               'ci95':np.quantile(distribution,[.025,.975]).tolist(),
               'test':'centered_paired_block_bootstrap' if temporal else 'centered_paired_bootstrap',
               'p_value':None, 'n_observations':n, 'block_length':length}
        involved = [m for m in (a,b) if m in method_results]
        if any(not method_results[m]['fidelity']['ranking_has_signal'] for m in involved):
            row['status'] = 'uninformative_constant_ranking'
        elif n < 30 or (temporal and n//length < 8):
            row['status'] = 'insufficient_observations_or_blocks'
        elif np.all(distribution == 0) and observed == 0:
            row.update(status='identical', p_value=1.)
        elif np.std(distribution) < 1e-14:
            row['status'] = 'degenerate_bootstrap_distribution'
        else:
            centered = distribution-float(observed)
            p = (1+np.count_nonzero(np.abs(centered)>=abs(observed)-1e-14))/(len(distribution)+1)
            row.update(status='approximate_under_resampling_assumptions', p_value=float(p))
        rows.append(row)
    for method in method_results:
        for metric, control in [('top_minus_random','random'),('top_minus_bottom','bottom')]:
            append(metric,method,control,points[method][metric],distributions[method][metric])
    for a,b in itertools.combinations(method_results,2):
        append('difference_in_top_minus_random',a,b,points[a]['top_minus_random']-points[b]['top_minus_random'],
               distributions[a]['top_minus_random']-distributions[b]['top_minus_random'])
    return {'intervals':intervals, 'comparisons':holm(rows,'masking_comparisons_within_dataset_model'),
            'resamples':resamples, 'seed':seed, 'block_length':length,
            'scope':'fixed_model_and_rankings; held_out_observation_and_random_control_uncertainty',
            'assumptions': 'weakly_stationary_weakly_dependent_time_series' if temporal else 'independent_test_observations',
            'warning':'Intervals are marginal, not simultaneous. Holm correction is per dataset/model/family; no universal superiority claim.'}
