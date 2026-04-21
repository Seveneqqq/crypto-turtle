"""
Metrics calculation utilities for Random Forest model - log return only
-- identical to xgb_model/common/metrics.py --
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def pseudo_huber_loss(y_true, y_pred, delta=1.0):
    """
    Pseudo-Huber loss (mean over samples).
    L = delta^2 * (sqrt(1 + ((y_pred - y_true) / delta)^2) - 1)
    """
    residuals = y_pred - y_true
    return float(np.mean(delta ** 2 * (np.sqrt(1.0 + (residuals / delta) ** 2) - 1.0)))


def calculate_metrics(y_true_log, y_pred_log):
    """
    Calculate metrics on log returns.

    Args:
        y_true_log: True log returns
        y_pred_log: Predicted log returns

    Returns:
        dict: Metrics
    """
    metrics = {
        'mae':  float(mean_absolute_error(y_true_log, y_pred_log)),
        'rmse': float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        'mse':  float(mean_squared_error(y_true_log, y_pred_log)),
        'r2':   float(r2_score(y_true_log, y_pred_log)),
    }

    # Pseudo-Huber losses kept for comparability with XGBoost results
    metrics['pseudo_huber']       = pseudo_huber_loss(y_true_log, y_pred_log, delta=1.0)
    metrics['pseudo_huber_large'] = pseudo_huber_loss(y_true_log, y_pred_log, delta=10.0)

    errors = y_pred_log - y_true_log
    metrics['error_mean']   = float(np.mean(errors))
    metrics['error_std']    = float(np.std(errors))
    metrics['error_median'] = float(np.median(errors))
    metrics['true_mean']    = float(np.mean(y_true_log))
    metrics['pred_mean']    = float(np.mean(y_pred_log))

    # Directional accuracy: sign of log return = direction
    actual_dir = np.sign(y_true_log)
    pred_dir   = np.sign(y_pred_log)
    metrics['dir_acc'] = float(np.mean(actual_dir == pred_dir) * 100)

    return metrics


def calculate_threshold_metrics(y_true_log, y_pred_log, threshold):
    """
    Calculate metrics for samples where |y_pred_log| >= threshold.

    Mirrors model_package signal logic: predictions below threshold are
    'no signal' and are excluded from evaluation.  This gives a more
    realistic picture of model performance on samples that would actually
    generate a BUY or SELL signal.

    Args:
        y_true_log:  np.ndarray -- true log-returns (all samples)
        y_pred_log:  np.ndarray -- predicted log-returns (all samples)
        threshold:   float -- minimum |y_pred| to include in metrics

    Returns:
        dict with threshold metrics, or None if no samples pass the filter.
        Keys:
            threshold           -- the threshold value used
            n_above             -- number of samples with |y_pred| >= threshold
            pct_above           -- percentage of total samples that are above
            mae_thresh          -- MAE on above-threshold subset
            rmse_thresh         -- RMSE on above-threshold subset
            r2_thresh           -- R2 on above-threshold subset
            pseudo_huber_thresh      -- PH(d=1) on subset
            pseudo_huber_large_thresh-- PH(d=10) on subset
            dir_acc_thresh      -- directional accuracy on subset
            bias_buy            -- fraction of above-threshold preds that are BUY (> 0)
    """
    mask = np.abs(y_pred_log) >= threshold
    n_above = int(np.sum(mask))
    n_total = len(y_pred_log)

    if n_above == 0:
        return {
            'threshold':  float(threshold),
            'n_above':    0,
            'pct_above':  0.0,
            'mae_thresh': None,
            'rmse_thresh': None,
            'r2_thresh':  None,
            'pseudo_huber_thresh': None,
            'pseudo_huber_large_thresh': None,
            'dir_acc_thresh': None,
            'bias_buy': None,
        }

    yt = y_true_log[mask]
    yp = y_pred_log[mask]

    result = {
        'threshold':  float(threshold),
        'n_above':    n_above,
        'pct_above':  float(n_above / n_total * 100),
        'mae_thresh': float(mean_absolute_error(yt, yp)),
        'rmse_thresh': float(np.sqrt(mean_squared_error(yt, yp))),
        'r2_thresh':  float(r2_score(yt, yp)),
        'pseudo_huber_thresh':       pseudo_huber_loss(yt, yp, delta=1.0),
        'pseudo_huber_large_thresh': pseudo_huber_loss(yt, yp, delta=10.0),
        'dir_acc_thresh': float(np.mean(np.sign(yt) == np.sign(yp)) * 100),
        'bias_buy':   float(np.mean(yp > 0) * 100),
    }
    return result


def print_metrics_summary(metrics, thresh_metrics=None):
    """Print formatted metrics summary"""
    print("\n" + "=" * 70)
    print("METRICS SUMMARY")
    print("=" * 70)

    print("\nLOG RETURN METRICS (ALL SAMPLES):")
    print(f"  MAE:                   {metrics['mae']:.6f}")
    print(f"  RMSE:                  {metrics['rmse']:.6f}")
    print(f"  R2:                    {metrics['r2']:.4f}")
    print(f"  Pseudo-Huber (d=1):    {metrics['pseudo_huber']:.6f}")
    print(f"  Pseudo-Huber (d=10):   {metrics['pseudo_huber_large']:.6f}")

    print("\nERROR STATS:")
    print(f"  Mean:   {metrics['error_mean']:.6f}")
    print(f"  Median: {metrics['error_median']:.6f}")
    print(f"  Std:    {metrics['error_std']:.6f}")

    if 'dir_acc' in metrics:
        print("\nDIRECTIONAL (ALL SAMPLES):")
        print(f"  Accuracy: {metrics['dir_acc']:.2f}%")

    if thresh_metrics is not None and thresh_metrics.get('n_above', 0) > 0:
        T   = thresh_metrics['threshold']
        n   = thresh_metrics['n_above']
        pct = thresh_metrics['pct_above']
        print(f"\nTHRESHOLD METRICS (|y_pred| >= {T:.4f}):")
        print(f"  Samples above threshold: {n}  ({pct:.1f}% of all)")
        # n_total is in thresh_metrics itself via pct_above calculation
        print(f"  MAE:                   {thresh_metrics['mae_thresh']:.6f}")
        print(f"  RMSE:                  {thresh_metrics['rmse_thresh']:.6f}")
        print(f"  R2:                    {thresh_metrics['r2_thresh']:.4f}")
        print(f"  Pseudo-Huber (d=1):    {thresh_metrics['pseudo_huber_thresh']:.6f}")
        print(f"  Pseudo-Huber (d=10):   {thresh_metrics['pseudo_huber_large_thresh']:.6f}")
        print(f"  Dir Acc:               {thresh_metrics['dir_acc_thresh']:.2f}%")
        print(f"  Bias BUY:              {thresh_metrics['bias_buy']:.1f}%")
    elif thresh_metrics is not None and thresh_metrics.get('n_above', 0) == 0:
        print(f"\nTHRESHOLD METRICS: no predictions above threshold {thresh_metrics['threshold']:.4f}")


def interpret_metrics(metrics, thresh_metrics=None):
    """Provide interpretation of metrics quality"""
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    r2 = metrics['r2']
    if r2 > 0.1:
        print(f"  [+] Good fit: R2 = {r2:.4f}")
    elif r2 > 0.05:
        print(f"  [o] Moderate fit: R2 = {r2:.4f}")
    else:
        print(f"  [-] Weak fit: R2 = {r2:.4f}")

    dir_acc = metrics.get('dir_acc', 0)
    if dir_acc > 0:
        if dir_acc > 55:
            print(f"  [+] Good direction (all): {dir_acc:.1f}%")
        elif dir_acc > 50:
            print(f"  [o] Slight edge (all): {dir_acc:.1f}%")
        else:
            print(f"  [-] No edge (all): {dir_acc:.1f}%")

    if thresh_metrics is not None and thresh_metrics.get('dir_acc_thresh') is not None:
        da = thresh_metrics['dir_acc_thresh']
        T  = thresh_metrics['threshold']
        pct = thresh_metrics['pct_above']
        label = f"above threshold {T:.4f} ({pct:.1f}% of samples)"
        if da > 55:
            print(f"  [+] Good direction ({label}): {da:.1f}%")
        elif da > 50:
            print(f"  [o] Slight edge ({label}): {da:.1f}%")
        else:
            print(f"  [-] No edge ({label}): {da:.1f}%")
