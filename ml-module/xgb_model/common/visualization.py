"""
Visualization utilities for XGBoost model - log return only
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_test_results(y_true_log, y_pred_log, output_dir, horizon):
    """
    Create test result visualizations for log returns.

    Args:
        y_true_log: True log returns
        y_pred_log: Predicted log returns
        output_dir: Output directory
        horizon: Prediction horizon
    """
    print("\nGenerating visualizations...")
    n = len(y_true_log)

    # 1. Full time series
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(y_true_log, 'b-', label='Actual', linewidth=1.0, alpha=0.7)
    ax.plot(y_pred_log, 'r-', label='Predicted', linewidth=1.0, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Log Return')
    ax.set_title(f'Log Returns: Actual vs Predicted (n={n}, horizon t+{horizon})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'log_returns_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ log_returns_timeseries.png")

    # 2. Scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true_log, y_pred_log, alpha=0.3, s=10, color='green')
    lo = min(y_true_log.min(), y_pred_log.min())
    hi = max(y_true_log.max(), y_pred_log.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual Log Return')
    ax.set_ylabel('Predicted Log Return')
    ax.set_title(f'Log Returns: Predicted vs Actual (n={n})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'log_returns_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ log_returns_scatter.png")

    # 3. Error distribution
    errors = y_pred_log - y_true_log
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=60, alpha=0.7, edgecolor='black', color='orange')
    ax.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero')
    ax.axvline(x=float(np.mean(errors)), color='g', linestyle='--', lw=2,
               label=f'Mean: {np.mean(errors):.6f}')
    ax.set_xlabel('Error (log return)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ error_distribution.png")

    # 4. Combined 2x2
    n_grid = min(1000, n)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    axes[0, 0].plot(y_true_log[:n_grid], 'b-', label='Actual', linewidth=1.0, alpha=0.7)
    axes[0, 0].plot(y_pred_log[:n_grid], 'r-', label='Predicted', linewidth=1.0, alpha=0.7)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
    axes[0, 0].set_title(f'Time Series (n={n_grid})')
    axes[0, 0].set_ylabel('Log Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(y_true_log, y_pred_log, alpha=0.3, s=10, color='green')
    axes[0, 1].plot([lo, hi], [lo, hi], 'r--', lw=2)
    axes[0, 1].set_title(f'Scatter (n={n})')
    axes[0, 1].set_xlabel('Actual')
    axes[0, 1].set_ylabel('Predicted')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(errors, bins=60, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].set_xlabel('Error')
    axes[1, 0].grid(True, alpha=0.3)

    # Directional accuracy by sign
    actual_dir = np.sign(y_true_log)
    pred_dir = np.sign(y_pred_log)
    correct = (actual_dir == pred_dir).astype(float)
    # Rolling accuracy
    window = min(500, n // 4)
    if window > 10:
        rolling_acc = pd.Series(correct).rolling(window).mean() * 100 if False else \
            np.convolve(correct, np.ones(window) / window, mode='valid') * 100
        axes[1, 1].plot(rolling_acc, color='purple', linewidth=1.0)
        axes[1, 1].axhline(y=50, color='r', linestyle='--', lw=1.5, alpha=0.7)
        axes[1, 1].set_title(f'Rolling Dir Accuracy (window={window})')
        axes[1, 1].set_xlabel('Sample')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, f'Dir Acc: {correct.mean()*100:.2f}%',
                        transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=14)
        axes[1, 1].set_title('Directional Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ combined_results.png")


def plot_feature_importance(model, feature_names, output_dir, top_n=30):
    """Plot XGBoost feature importance (supports both Booster and XGBRegressor)"""
    import xgboost as xgb
    if isinstance(model, xgb.Booster):
        scores = model.get_score(importance_type='gain')
        # get_score keys are 'f0','f1',... when DMatrix was created without feature_names.
        # Map positional keys to actual names before lookup.
        remapped = {}
        for k, v in scores.items():
            if k.startswith('f') and k[1:].isdigit():
                idx = int(k[1:])
                if idx < len(feature_names):
                    remapped[feature_names[idx]] = v
            else:
                remapped[k] = v
        importance = np.array([remapped.get(f, 0.0) for f in feature_names])
    else:
        importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
    y_pos = np.arange(top_n)
    ax.barh(y_pos, importance[indices][::-1], alpha=0.7, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=9)
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ feature_importance.png")
