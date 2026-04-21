"""
XGBoost Model Testing
"""
import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from xgb_config import XGBConfig
from common.data_utils import load_test_data_only
from common.id_utils import get_next_test_id, extract_model_id, extract_data_id
from common.metrics import calculate_metrics, print_metrics_summary, interpret_metrics
from common.sample_weighting import resolve_signal_threshold
from common.visualization import plot_test_results, plot_feature_importance
from common.log_utils import (
    log_run_header, log_model_info, log_test_run_info, log_test_complete,
)

warnings.filterwarnings('ignore')


def load_model(model_dir):
    """Load trained XGBoost model and configuration"""
    config_path = os.path.join(model_dir, 'model_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing model_config.json in {model_dir}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, 'xgb_model.json'))

    feature_scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))

    target_scaler = None
    target_scaler_path = os.path.join(model_dir, 'target_scaler.pkl')
    if os.path.exists(target_scaler_path):
        target_scaler = joblib.load(target_scaler_path)

    return model, config, feature_scaler, target_scaler


def auto_detect_test_path(config):
    """
    Auto-detect test data path from training configuration.

    Search order:
      1. Directory used during training (if it contains metadata.json + test.csv)
      2. prepared_data/ by data_id
      3. data_preparation/prepared_data/ by data_id (universal prepared dirs)
      4. rf_model/rf_prepared_data/ by data_id (RF prepared dirs, same format)
    """
    train_path = config.get('train_path', '')
    data_id    = config.get('data_id', 'UNKNOWN')

    # Strategy 1: model was trained on a prepared directory that has test.csv
    if os.path.isdir(train_path) and os.path.exists(
            os.path.join(train_path, 'metadata.json')):
        if os.path.exists(os.path.join(train_path, 'test.csv')):
            return train_path

    if data_id in ('RAW', 'UNKNOWN'):
        return None

    def _search_prep_dir(base):
        """Search base directory for a subdirectory starting with data_id."""
        if not os.path.exists(base):
            return None
        matching = [
            d for d in os.listdir(base)
            if d.startswith(data_id) and os.path.isdir(os.path.join(base, d))
        ]
        if not matching:
            return None
        candidate = os.path.join(base, sorted(matching)[-1])
        if os.path.exists(os.path.join(candidate, 'test.csv')):
            return candidate
        return None

    project_root = os.path.dirname(XGBConfig.BASE_DIR)   # claude_xgb/

    # Strategy 2: prepared_data (XGBoost's own prepared directory)
    result = _search_prep_dir(XGBConfig.PREPARED_DATA_DIR)
    if result:
        return result

    # Strategy 3: data_preparation/prepared_data (universal prepared directories)
    result = _search_prep_dir(
        os.path.join(project_root, 'data_preparation', 'prepared_data')
    )
    if result:
        return result

    # Strategy 4: rf_model/rf_prepared_data (RF prepared dirs, same format)
    result = _search_prep_dir(
        os.path.join(project_root, 'rf_model', 'rf_prepared_data')
    )
    if result:
        return result

    return None


def _load_csv_for_test(csv_path, config, feature_scaler):
    """
    Load a raw CSV file as test data — uses the entire file, no train/val split.
    Applies universal feature engineering and the model's feature_scaler.
    Returns X_test, y_test (original log-return space), metadata dict.
    """
    # Import from universal FE_ via the local bridge
    from feature_engineering import prepare_features
    from common.data_utils import flatten_features, create_log_return_target

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from CSV")

    df = prepare_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"After feature engineering: {len(df)} samples")

    features      = config['features']
    lookback      = config['lookback']
    horizon       = config['horizon']
    flat_method   = config['flattening_method']

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"CSV missing features required by model: {missing}")

    df[features] = feature_scaler.transform(df[features])

    X_flat, flat_features = flatten_features(df, features, lookback, method=flat_method)
    y_all = create_log_return_target(df, lookback, horizon)
    X_flat = X_flat.iloc[lookback:].reset_index(drop=True).dropna()

    n = min(len(X_flat), len(y_all))
    X_test = X_flat.iloc[:n].values
    y_test = y_all[:n]

    print(f"Test shape: {X_test.shape}")

    metadata = {
        'data_id':          'CSV',
        'features':         features,
        'flat_features':    flat_features,
        'n_features':       len(features),
        'n_flat_features':  len(flat_features),
        'lookback':         lookback,
        'horizon':          horizon,
        'flattening_method': flat_method,
        'source_file':      os.path.basename(csv_path),
    }

    return X_test, y_test, metadata


def test_model(model_dir=None, test_path=None):
    """Test XGBoost model on test data"""

    if model_dir is None:
        if not os.path.exists(XGBConfig.MODELS_DIR):
            raise FileNotFoundError(f"Models directory not found: {XGBConfig.MODELS_DIR}")

        model_dirs = [d for d in os.listdir(XGBConfig.MODELS_DIR)
                      if d.startswith('X') and os.path.isdir(os.path.join(XGBConfig.MODELS_DIR, d))]

        if not model_dirs:
            raise FileNotFoundError(f"No trained models found in {XGBConfig.MODELS_DIR}")

        model_dir = os.path.join(XGBConfig.MODELS_DIR, sorted(model_dirs)[-1])
        print(f"Auto-selected latest model: {os.path.basename(model_dir)}")

    if not os.path.isabs(model_dir) and not os.path.exists(model_dir):
        model_dir = os.path.join(XGBConfig.MODELS_DIR, model_dir)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_id = extract_model_id(model_dir)

    log_run_header("XGBOOST MODEL TESTING", newline=True)

    print(f"\nLoading model: {model_dir}")
    model, config, feature_scaler, target_scaler = load_model(model_dir)

    log_model_info(model_id, config, target_scaler)

    if test_path is None:
        test_path = auto_detect_test_path(config)
        if test_path is None:
            raise FileNotFoundError(
                f"Cannot auto-detect test data for model {model_id}.\n"
                f"Please specify: --test_path <path>"
            )
        print(f"\n✓ Auto-detected test data: {test_path}")
    else:
        print(f"\nTest data: {test_path}")

    print("\nLoading test data...")
    if os.path.isdir(test_path):
        data, metadata = load_test_data_only(
            test_path,
            flattening_method=config['flattening_method'],
            feature_set=config.get('feature_set'),
            custom_features=config.get('custom_features'),
            lookback=config['lookback'],
            horizon=config['horizon']
        )
        X_test = data['X_test']
        y_test = data['y_test']
    elif test_path.endswith('.csv'):
        X_test, y_test, metadata = _load_csv_for_test(
            test_path, config, feature_scaler
        )
    else:
        raise ValueError(f"Invalid test_path: {test_path}. Expected directory or .csv file")

    # Setup output
    test_id = get_next_test_id(XGBConfig.LOGS_DIR)
    data_id = extract_data_id(metadata) if metadata else config['data_id']
    date_short = datetime.now().strftime('%d%m')
    run_name = f"{test_id}_test_results_{model_id}_{data_id}_{date_short}"
    output_dir = os.path.join(XGBConfig.LOGS_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)

    log_test_run_info(test_id, len(X_test), output_dir)

    # Predictions
    log_run_header("GENERATING PREDICTIONS", newline=True)

    saved_iter = model.attr('saved_iteration')
    if saved_iter is not None:
        n_saved = int(saved_iter)
        iter_range = (0, n_saved)
    else:
        iter_range = (0, model.num_boosted_rounds())

    expected_cols = config.get('n_flat_features')
    if expected_cols is not None and X_test.shape[1] != expected_cols:
        print(f"[SHAPE WARNING] X_test cols={X_test.shape[1]}, "
              f"model expects {expected_cols} flat features — mismatch may cause errors")

    dtest = xgb.DMatrix(X_test)
    y_pred_scaled = model.predict(dtest, iteration_range=iter_range)

    if target_scaler is not None:
        y_pred_log = target_scaler.inverse_transform(y_pred_scaled)
    else:
        y_pred_log = y_pred_scaled

    print(f"Generated {len(y_pred_log)} predictions (iter_range={iter_range})")

    # Metrics
    log_run_header("EVALUATION METRICS", newline=True)

    # Resolve signal threshold for thresholded metrics
    signal_threshold_enabled = config.get('signal_threshold_enabled', XGBConfig.SIGNAL_THRESHOLD_ENABLED)
    resolved_threshold = None
    if signal_threshold_enabled:
        saved_threshold = config.get('signal_threshold_resolved')
        if saved_threshold is not None:
            resolved_threshold = float(saved_threshold)
            print(f"\nSignal threshold: {resolved_threshold:.6f} (from model config)")
        else:
            # Fallback for older models without a saved threshold.
            # y_train is unavailable here, so 'empirical' mode cannot be used safely
            # (it would compute the percentile from test data — data leakage).
            # Force 'fixed' mode regardless of what was recorded in config.
            _saved_mode = config.get('signal_threshold_mode', XGBConfig.SIGNAL_THRESHOLD_MODE)
            _fixed_value = config.get('signal_threshold_value', XGBConfig.SIGNAL_THRESHOLD_VALUE)
            if _saved_mode == 'empirical':
                print(
                    f"\n[WARN] Model used signal_threshold_mode='empirical' but y_train is "
                    f"unavailable during testing — falling back to fixed threshold "
                    f"({_fixed_value}) to avoid data leakage."
                )
            resolved_threshold = resolve_signal_threshold(
                y_test,       # ignored by 'fixed' mode
                mode='fixed',
                fixed_value=_fixed_value,
            )
            print(f"\nSignal threshold: {resolved_threshold:.6f} (fallback fixed — model config missing)")

    metrics = calculate_metrics(y_test, y_pred_log, threshold=resolved_threshold)
    print_metrics_summary(metrics)
    interpret_metrics(metrics)

    # Visualizations
    log_run_header("GENERATING VISUALIZATIONS", newline=True)

    plot_test_results(y_test, y_pred_log, output_dir, config['horizon'])

    if config.get('flat_features'):
        plot_feature_importance(model, config['flat_features'], output_dir, top_n=30)

    # Save results
    log_run_header("SAVING RESULTS", newline=True)

    pd.DataFrame({
        'actual_log': y_test,
        'predicted_log': y_pred_log,
        'error': y_pred_log - y_test,
    }).to_csv(os.path.join(output_dir, 'results.csv'), index=False)
    print("  ✓ results.csv")

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("  ✓ metrics.json")

    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("XGBOOST MODEL TEST SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test ID:    {test_id}\n")
        f.write(f"Model ID:   {model_id}\n")
        f.write(f"Data ID:    {data_id}\n")
        f.write(f"Symbol:     {config['symbol'].upper()}\n")
        f.write(f"Date:       {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
        f.write(f"Horizon:    t+{config['horizon']}\n")
        f.write(f"Samples:    {len(y_test)}\n\n")
        f.write("LOG RETURN METRICS:\n")
        f.write(f"  MAE:                   {metrics['mae']:.6f}\n")
        f.write(f"  RMSE:                  {metrics['rmse']:.6f}\n")
        f.write(f"  R²:                    {metrics['r2']:.4f}\n")
        f.write(f"  Pseudo-Huber (δ=1):    {metrics['pseudo_huber']:.6f}\n")
        f.write(f"  Pseudo-Huber (δ=10):   {metrics['pseudo_huber_large']:.6f}\n")
        f.write(f"  Dir Acc:               {metrics['dir_acc']:.2f}%\n")
        if 't_mae' in metrics:
            f.write(f"\nTHRESHOLDED METRICS (|y| ≥ {metrics['t_threshold']:.6f},"
                    f" n={metrics['t_n_samples']}):\n")
            f.write(f"  t_MAE:                 {metrics['t_mae']:.6f}\n")
            f.write(f"  t_RMSE:                {metrics['t_rmse']:.6f}\n")
            f.write(f"  t_R²:                  {metrics['t_r2']:.4f}\n")
            f.write(f"  t_Pseudo-Huber (δ=1):  {metrics['t_pseudo_huber']:.6f}\n")
            f.write(f"  t_Pseudo-Huber (δ=10): {metrics['t_pseudo_huber_large']:.6f}\n")
            f.write(f"  t_Dir Acc:             {metrics['t_dir_acc']:.2f}%\n")
    print("  ✓ summary.txt")

    log_test_complete(test_id, model_id, len(y_test), metrics, output_dir)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test XGBoost model')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    args = parser.parse_args()

    try:
        test_model(model_dir=args.model_dir, test_path=args.test_path)
    except (FileNotFoundError, Exception) as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        if not isinstance(e, FileNotFoundError):
            import traceback
            traceback.print_exc()
        raise
