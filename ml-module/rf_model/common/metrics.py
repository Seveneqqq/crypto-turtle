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


def print_metrics_summary(metrics):
    """Print formatted metrics summary"""
    print("\n" + "=" * 70)
    print("METRICS SUMMARY")
    print("=" * 70)

    print("\nLOG RETURN METRICS:")
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
        print("\nDIRECTIONAL:")
        print(f"  Accuracy: {metrics['dir_acc']:.2f}%")


def interpret_metrics(metrics):
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
            print(f"  [+] Good direction: {dir_acc:.1f}%")
        elif dir_acc > 50:
            print(f"  [o] Slight edge: {dir_acc:.1f}%")
        else:
            print(f"  [-] No edge: {dir_acc:.1f}%")
