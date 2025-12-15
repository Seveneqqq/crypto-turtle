"""
Metrics calculation utilities
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred, price_scaler, use_log_prices=False):
    """Calculate comprehensive metrics for price predictions"""
    if use_log_prices:
        y_true_log = price_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_log = price_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_true_price = np.exp(y_true_log)
        y_pred_price = np.exp(y_pred_log)
    else:
        y_true_price = price_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_price = price_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    metrics = {
        'mae': mean_absolute_error(y_true_price, y_pred_price),
        'rmse': np.sqrt(mean_squared_error(y_true_price, y_pred_price)),
        'mse': mean_squared_error(y_true_price, y_pred_price),
        'r2': r2_score(y_true_price, y_pred_price)
    }

    mape = np.mean(np.abs((y_true_price - y_pred_price) / (y_true_price + 1e-8))) * 100
    metrics['mape'] = mape if np.isfinite(mape) else 999.99

    errors = y_pred_price - y_true_price
    metrics['error_mean'] = np.mean(errors)
    metrics['error_std'] = np.std(errors)
    metrics['error_median'] = np.median(errors)
    metrics['true_mean'] = np.mean(y_true_price)
    metrics['pred_mean'] = np.mean(y_pred_price)

    return metrics, y_true_price, y_pred_price


def calculate_directional_accuracy(y_true_price, y_pred_price, current_prices):
    """Calculate directional prediction accuracy"""
    if len(current_prices) == 0 or np.all(current_prices == 0):
        return 0.0

    actual_dir = np.sign(y_true_price - current_prices)
    pred_dir = np.sign(y_pred_price - current_prices)
    return np.mean(actual_dir == pred_dir) * 100


def print_metrics_summary(metrics, dir_acc=None):
    """Print formatted metrics summary"""
    print("\nPRICE METRICS:")
    print(f"  MAE:   ${metrics['mae']:,.2f}")
    print(f"  RMSE:  ${metrics['rmse']:,.2f}")
    print(f"  MAPE:  {metrics['mape']:.2f}%")
    print(f"  R²:    {metrics['r2']:.4f}")

    print("\nERROR STATS:")
    print(f"  Mean:   ${metrics['error_mean']:,.2f}")
    print(f"  Median: ${metrics['error_median']:,.2f}")
    print(f"  Std:    ${metrics['error_std']:,.2f}")

    print("\nPRICE STATS:")
    print(f"  Actual:    ${metrics['true_mean']:,.2f}")
    print(f"  Predicted: ${metrics['pred_mean']:,.2f}")

    if dir_acc is not None and dir_acc > 0:
        print("\nDIRECTIONAL:")
        print(f"  Accuracy: {dir_acc:.2f}%")


def interpret_metrics(metrics, dir_acc=None):
    """Provide interpretation of metrics"""
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if metrics['mape'] < 1.0:
        print("  ✓ EXCELLENT: MAPE < 1%")
    elif metrics['mape'] < 2.0:
        print("  ✓ GOOD: MAPE 1-2%")
    elif metrics['mape'] < 5.0:
        print("  ○ ACCEPTABLE: MAPE 2-5%")
    else:
        print("  ✗ POOR: MAPE > 5%")

    if metrics['r2'] > 0.5:
        print(f"  ✓ Strong: R² = {metrics['r2']:.3f}")
    elif metrics['r2'] > 0.2:
        print(f"  ○ Moderate: R² = {metrics['r2']:.3f}")
    else:
        print(f"  ✗ Weak: R² = {metrics['r2']:.3f}")

    if dir_acc is not None and dir_acc > 0:
        if dir_acc > 55:
            print(f"  ✓ Good direction: {dir_acc:.1f}%")
        elif dir_acc > 50:
            print(f"  ○ Slight edge: {dir_acc:.1f}%")
        else:
            print(f"  ✗ No edge: {dir_acc:.1f}%")