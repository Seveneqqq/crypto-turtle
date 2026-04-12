"""
Metrics calculation utilities for XGBoost model - log return only
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


def calculate_metrics(y_true_log, y_pred_log, threshold=None):
    """
    Calculate metrics on log returns.

    Args:
        y_true_log: True log returns
        y_pred_log: Predicted log returns
        threshold:  Optional signal threshold. When > 0, computes additional
                    thresholded metrics (t_ prefix) on samples where
                    |y_true| >= threshold.

    Returns:
        dict: Metrics (standard + optional thresholded)
    """
    metrics = {
        'mae':  float(mean_absolute_error(y_true_log, y_pred_log)),
        'rmse': float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        'mse':  float(mean_squared_error(y_true_log, y_pred_log)),
        'r2':   float(r2_score(y_true_log, y_pred_log)),
    }

    # Pseudo-Huber losses
    # Standard delta=1.0  → reg:pseudohubererror
    # Large   delta=10.0  → reg:pseudohubererror_large (huber_slope=10)
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

    # Thresholded metrics (computed on samples where |y_true| >= threshold)
    if threshold is not None and threshold > 0:
        mask = np.abs(y_true_log) >= threshold
        n_above = int(np.sum(mask))
        metrics['t_n_samples']   = n_above
        metrics['t_pct_samples'] = float(n_above / max(len(y_true_log), 1) * 100)
        metrics['t_threshold']   = float(threshold)

        if n_above >= 10:
            yt = y_true_log[mask]
            yp = y_pred_log[mask]
            metrics['t_mae']  = float(mean_absolute_error(yt, yp))
            metrics['t_rmse'] = float(np.sqrt(mean_squared_error(yt, yp)))
            metrics['t_r2']   = float(r2_score(yt, yp))
            metrics['t_pseudo_huber']       = pseudo_huber_loss(yt, yp, delta=1.0)
            metrics['t_pseudo_huber_large'] = pseudo_huber_loss(yt, yp, delta=10.0)
            t_actual_dir = np.sign(yt)
            t_pred_dir   = np.sign(yp)
            metrics['t_dir_acc'] = float(np.mean(t_actual_dir == t_pred_dir) * 100)

    return metrics


def print_metrics_summary(metrics):
    """Print formatted metrics summary"""
    print("\n" + "=" * 70)
    print("METRICS SUMMARY")
    print("=" * 70)

    print("\nLOG RETURN METRICS:")
    print(f"  MAE:                   {metrics['mae']:.6f}")
    print(f"  RMSE:                  {metrics['rmse']:.6f}")
    print(f"  R²:                    {metrics['r2']:.4f}")
    print(f"  Pseudo-Huber (δ=1):    {metrics['pseudo_huber']:.6f}")
    print(f"  Pseudo-Huber (δ=10):   {metrics['pseudo_huber_large']:.6f}")

    print("\nERROR STATS:")
    print(f"  Mean:   {metrics['error_mean']:.6f}")
    print(f"  Median: {metrics['error_median']:.6f}")
    print(f"  Std:    {metrics['error_std']:.6f}")

    if 'dir_acc' in metrics:
        print("\nDIRECTIONAL:")
        print(f"  Accuracy: {metrics['dir_acc']:.2f}%")

    if 't_mae' in metrics:
        print(f"\nTHRESHOLDED METRICS (|y| ≥ {metrics['t_threshold']:.6f}, "
              f"n={metrics['t_n_samples']} / {metrics['t_pct_samples']:.1f}%):")
        print(f"  MAE:                   {metrics['t_mae']:.6f}")
        print(f"  RMSE:                  {metrics['t_rmse']:.6f}")
        print(f"  R²:                    {metrics['t_r2']:.4f}")
        print(f"  Pseudo-Huber (δ=1):    {metrics['t_pseudo_huber']:.6f}")
        print(f"  Pseudo-Huber (δ=10):   {metrics['t_pseudo_huber_large']:.6f}")
        print(f"  Dir Acc:               {metrics['t_dir_acc']:.2f}%")
    elif 't_n_samples' in metrics:
        print(f"\nTHRESHOLDED: insufficient samples above threshold "
              f"(n={metrics['t_n_samples']}, need ≥ 10)")


def interpret_metrics(metrics):
    """Provide interpretation of metrics quality"""
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    r2 = metrics['r2']
    if r2 > 0.1:
        print(f"  ✓ Good fit: R² = {r2:.4f}")
    elif r2 > 0.05:
        print(f"  ○ Moderate fit: R² = {r2:.4f}")
    else:
        print(f"  ✗ Weak fit: R² = {r2:.4f}")

    dir_acc = metrics.get('dir_acc', 0)
    if dir_acc > 0:
        if dir_acc > 55:
            print(f"  ✓ Good direction: {dir_acc:.1f}%")
        elif dir_acc > 50:
            print(f"  ○ Slight edge: {dir_acc:.1f}%")
        else:
            print(f"  ✗ No edge: {dir_acc:.1f}%")

    # Thresholded interpretation
    t_dir_acc = metrics.get('t_dir_acc')
    if t_dir_acc is not None:
        label = f"(|y| ≥ {metrics['t_threshold']:.4f})"
        if t_dir_acc > 60:
            print(f"  ✓ Strong thresholded direction {label}: {t_dir_acc:.1f}%")
        elif t_dir_acc > 55:
            print(f"  ✓ Good thresholded direction {label}: {t_dir_acc:.1f}%")
        elif t_dir_acc > 50:
            print(f"  ○ Slight thresholded edge {label}: {t_dir_acc:.1f}%")
        else:
            print(f"  ✗ No thresholded edge {label}: {t_dir_acc:.1f}%")
