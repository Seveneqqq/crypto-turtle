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
from common.visualization import plot_test_results, plot_feature_importance

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
    """Auto-detect test data path from training configuration"""
    train_path = config.get('train_path', '')
    data_id = config.get('data_id', 'UNKNOWN')

    if os.path.isdir(train_path) and os.path.exists(os.path.join(train_path, 'metadata.json')):
        return train_path

    if data_id not in ['RAW', 'UNKNOWN']:
        prep_base = XGBConfig.PREPARED_DATA_DIR
        if os.path.exists(prep_base):
            matching = [d for d in os.listdir(prep_base)
                        if d.startswith(data_id) and os.path.isdir(os.path.join(prep_base, d))]
            if matching:
                return os.path.join(prep_base, sorted(matching)[-1])

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

    print("\n" + "=" * 80)
    print("XGBOOST MODEL TESTING")
    print("=" * 80)

    print(f"\nLoading model: {model_dir}")
    model, config, feature_scaler, target_scaler = load_model(model_dir)

    print(f"Model ID:   {model_id}")
    print(f"Data ID:    {config['data_id']}")
    print(f"Symbol:     {config['symbol'].upper()}")
    print(f"Horizon:    t+{config['horizon']}")
    print(f"Features:   {config['n_features']} → {config['n_flat_features']} flat")
    print(f"Flattening: {config['flattening_method']}")
    print(f"Target scaler: {'ENABLED (' + target_scaler.method + ')' if target_scaler else 'DISABLED'}")

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

    print(f"\nTest ID:    {test_id}")
    print(f"Samples:    {len(X_test)}")
    print(f"Output:     {output_dir}")

    # Predictions
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)

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
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)

    metrics = calculate_metrics(y_test, y_pred_log)
    print_metrics_summary(metrics)
    interpret_metrics(metrics)

    # Visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_test_results(y_test, y_pred_log, output_dir, config['horizon'])

    if config.get('flat_features'):
        plot_feature_importance(model, config['flat_features'], output_dir, top_n=30)

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

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
    print("  ✓ summary.txt")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"Test ID:  {test_id}")
    print(f"Model ID: {model_id}")
    print(f"Samples:  {len(y_test)}")
    print(f"MAE:      {metrics['mae']:.6f}")
    print(f"R²:       {metrics['r2']:.4f}")
    print(f"PH(δ=1):  {metrics['pseudo_huber']:.6f}")
    print(f"PH(δ=10): {metrics['pseudo_huber_large']:.6f}")
    print(f"Dir Acc:  {metrics['dir_acc']:.2f}%")
    print(f"Output:   {output_dir}")
    print("=" * 80)

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
