"""
Visualization utilities for LSTM pipeline
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, r2_score, mean_absolute_error
import optuna
from optuna.trial import TrialState


def plot_training_curves(train_losses, val_losses, best_val_loss, output_path):
    """Plot training and validation loss curves"""
    if len(train_losses) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train', alpha=0.8)
    ax.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Val', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(f'Training Curves - Best: {best_val_loss:.6f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_test_results(y_true_price, y_pred_price, current_prices, output_dir, horizon):
    """Create comprehensive test result visualizations"""
    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Time series
    n = min(500, len(y_true_price))
    x = np.arange(n)
    axes[0, 0].plot(x, y_true_price[:n], 'b-', label='Actual', linewidth=2, alpha=0.8)
    axes[0, 0].plot(x, y_pred_price[:n], 'r-', label='Predicted', linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].set_title(f'Actual vs Predicted (first {n} samples, horizon t+{horizon})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    r2 = r2_score(y_true_price[:n], y_pred_price[:n])
    mae = mean_absolute_error(y_true_price[:n], y_pred_price[:n])
    axes[0, 0].text(0.02, 0.98, f'R² = {r2:.4f}\nMAE = ${mae:.2f}',
                    transform=axes[0, 0].transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # 2. Scatter plot
    axes[0, 1].scatter(y_true_price, y_pred_price, alpha=0.3, s=10, color='green')
    min_val = min(y_true_price.min(), y_pred_price.min())
    max_val = max(y_true_price.max(), y_pred_price.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect', alpha=0.7)
    axes[0, 1].set_xlabel('Actual Price ($)')
    axes[0, 1].set_ylabel('Predicted Price ($)')
    axes[0, 1].set_title('Predicted vs Actual')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Error distribution
    errors = y_pred_price - y_true_price
    axes[1, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero')
    axes[1, 0].axvline(x=np.mean(errors), color='g', linestyle='--', lw=2,
                       label=f'Mean: ${np.mean(errors):.2f}')
    axes[1, 0].set_xlabel('Error ($)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Percentage error
    pct_errors = (errors / (y_true_price + 1e-8)) * 100
    axes[1, 1].hist(pct_errors, bins=50, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero')
    axes[1, 1].axvline(x=np.mean(pct_errors), color='g', linestyle='--', lw=2,
                       label=f'Mean: {np.mean(pct_errors):.2f}%')
    axes[1, 1].set_xlabel('Error (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Percentage Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ results.png")

    # Full time series
    fig, ax = plt.subplots(figsize=(20, 8))
    x_full = np.arange(len(y_true_price))
    ax.plot(x_full, y_true_price, 'b-', label='Actual', lw=1.5, alpha=0.7)
    ax.plot(x_full, y_pred_price, 'r-', label='Predicted', lw=1.5, alpha=0.7)
    ax.fill_between(x_full, y_true_price, y_pred_price, alpha=0.2, color='gray')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Price ($)')
    ax.set_title(f'Full Time Series (horizon t+{horizon}, n={len(y_true_price)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ timeseries.png")

    # Directional accuracy
    if len(current_prices) > 0 and not np.all(current_prices == 0):
        actual_dir = np.sign(y_true_price - current_prices)
        pred_dir = np.sign(y_pred_price - current_prices)
        correct = actual_dir == pred_dir
        dir_acc = np.mean(correct) * 100

        fig, axes = plt.subplots(2, 1, figsize=(18, 12))

        n_dir = min(200, len(actual_dir))
        x_dir = np.arange(n_dir)
        colors = ['green' if correct[i] else 'red' for i in range(n_dir)]

        axes[0].bar(x_dir, actual_dir[:n_dir], alpha=0.5, color='blue', label='Actual', width=0.8)
        axes[0].scatter(x_dir, pred_dir[:n_dir], c=colors, s=50, label='Predicted', marker='x', linewidths=2)
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Direction (1=UP, -1=DOWN)')
        axes[0].set_title(f'Directional Prediction (first {n_dir} samples)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([-1.5, 1.5])
        axes[0].text(0.02, 0.98, f'Accuracy: {dir_acc:.2f}%',
                     transform=axes[0].transAxes, va='top',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

        actual_bin = (actual_dir > 0).astype(int)
        pred_bin = (pred_dir > 0).astype(int)
        cm = confusion_matrix(actual_bin, pred_bin)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                    xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'],
                    cbar_kws={'label': 'Count'})
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        axes[1].set_title('Confusion Matrix - Direction')

        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / total * 100
                axes[1].text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                             ha='center', va='center', fontsize=10, color='red')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'directional.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ directional.png")

        return dir_acc

    return None


def plot_optuna_results(study, output_dir):
    """Create Optuna optimization visualizations"""
    print("\nGenerating plots...")

    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'history.png'), dpi=150)
        plt.close()
        print("  ✓ history.png")
    except:
        pass

    try:
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'importances.png'), dpi=150)
        plt.close()
        print("  ✓ importances.png")
    except:
        pass

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    losses = [t.value for t in completed]

    plt.figure(figsize=(10, 6))
    plt.hist(losses, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(study.best_value, color='r', linestyle='--', lw=2,
               label=f'Best: {study.best_value:.6f}')
    plt.xlabel('Validation Loss')
    plt.ylabel('Frequency')
    plt.title('Trial Results Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution.png'), dpi=150)
    plt.close()
    print("  ✓ distribution.png")