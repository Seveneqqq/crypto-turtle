"""
LSTM Price Prediction Testing
"""
import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm

from common.id_utils import get_next_test_id, extract_model_id, extract_data_id
from common.metrics import (calculate_metrics, calculate_directional_accuracy,
                             print_metrics_summary, interpret_metrics)
from common.visualization import plot_test_results

warnings.filterwarnings('ignore')


def _read_use_log_prices(config_or_metadata):
    """
    Read use_log_prices from model config or prepared-data metadata.
    Handles both formats:
      - Universal (new):  dict['scalers']['use_log_prices']
      - Legacy / config:  dict['use_log_prices']
    Defaults to True when key is absent.
    """
    scalers = config_or_metadata.get('scalers')
    if isinstance(scalers, dict) and 'use_log_prices' in scalers:
        return scalers['use_log_prices']
    return config_or_metadata.get('use_log_prices', True)


def load_test_data(test_path, model_config, lookback, horizon,
                   features, feature_scaler, price_scaler, use_log_prices):
    """
    Load test data for evaluation.

    Priority order for prepared directories:
      1. X_test_3d.npy / y_test_3d.npy  — pre-built 3-D sequences (DEFAULT)
      2. test_2d.npy                     — alternative, only when 3D files absent
      3. test.csv                        — alternative, only when 3D files absent

    Direct file arguments:
      - *.npy  — treated as 2-D flat array (sequences built on-the-fly)
      - *.csv  — raw OHLCV, features generated on-the-fly
    """
    is_prepared_dir = os.path.isdir(test_path)

    # ------------------------------------------------------------------ #
    # A) Prepared data directory
    # ------------------------------------------------------------------ #
    if is_prepared_dir:
        metadata_path = os.path.join(test_path, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise ValueError(f"Missing metadata.json in {test_path}")

        with open(metadata_path, 'r') as f:
            test_metadata = json.load(f)

        # --- Always prefer 3D sequence files ---
        x3d_path = os.path.join(test_path, 'X_test_3d.npy')
        y3d_path = os.path.join(test_path, 'y_test_3d.npy')

        if not os.path.exists(x3d_path) or not os.path.exists(y3d_path):
            raise FileNotFoundError(
                f"Required 3D sequence files not found in {test_path}:\n"
                f"  Missing: {'X_test_3d.npy' if not os.path.exists(x3d_path) else 'y_test_3d.npy'}\n"
                "These files are created by data_preparation/prepare_data.py."
            )

        X_test = np.load(x3d_path)
        y_test = np.load(y3d_path)
        print(f"Loaded 3D sequences: X={X_test.shape}, y={y_test.shape}")

        # --- Validate compatibility with model ---
        data_features = test_metadata.get('features', [])
        if data_features and data_features != features:
            print(f"Warning: Feature list mismatch — model has {len(features)}, "
                  f"data has {len(data_features)} features")

        data_lookback = test_metadata.get('lookback')
        data_horizon  = test_metadata.get('horizon')

        if data_lookback is not None and data_lookback != lookback:
            raise ValueError(
                f"Lookback mismatch: model={lookback}, prepared_data={data_lookback}")
        if data_horizon is not None and data_horizon != horizon:
            raise ValueError(
                f"Horizon mismatch: model={horizon}, prepared_data={data_horizon}")

        print(f"Test sequences: {len(X_test):,}")

        # --- current_prices: derive from X_test last time-step, last feature col
        # X_test shape: (n, lookback, n_features); last col is scaled close price
        # This gives the price at position [lookback-1] relative to each sample.
        # Note: the last feature column is NOT the close price — use zeros and
        # skip directional accuracy (close price is not stored in 3D sequences).
        current_prices = np.zeros(len(y_test))

        return X_test, y_test, current_prices, features

    # ------------------------------------------------------------------ #
    # B) Explicit 2D .npy file (no sequences — build on-the-fly)
    # ------------------------------------------------------------------ #
    if test_path.endswith('.npy'):
        print("Loading 2D NPY data...")
        data_2d = np.load(test_path)

        if data_2d.shape[1] != len(features) + 1:
            raise ValueError(
                f"Column count mismatch: expected {len(features) + 1} "
                f"(features + close), got {data_2d.shape[1]}")

        print(f"Loaded 2D data: {data_2d.shape}")
        print(f"Creating sequences (lookback={lookback}, horizon={horizon})...")

        n_samples = len(data_2d) - lookback - horizon + 1
        if n_samples <= 0:
            raise ValueError(
                f"Insufficient data: need at least {lookback + horizon} rows, got {len(data_2d)}")

        X_test        = np.zeros((n_samples, lookback, len(features)), dtype=np.float32)
        y_test        = np.zeros(n_samples, dtype=np.float32)
        current_prices = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            X_test[i]         = data_2d[i:i + lookback, :-1]
            current_prices[i] = data_2d[i + lookback - 1, -1]
            y_test[i]         = data_2d[i + lookback + horizon - 1, -1]

        print(f"Created {n_samples:,} sequences from 2D data")
        return X_test, y_test, current_prices, features

    # ------------------------------------------------------------------ #
    # C) Raw CSV — generate features + scale on-the-fly
    # ------------------------------------------------------------------ #
    if test_path.endswith('.csv'):
        df = pd.read_csv(test_path)
        print(f"Loaded {len(df):,} records (raw CSV)")

        from feature_engineering import prepare_features
        df = prepare_features(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"After feature engineering: {len(df):,} records")
        print(f"Price range: [${df['close'].min():,.2f}, ${df['close'].max():,.2f}]")

        available_features = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features in CSV: {missing}")
        if len(available_features) < len(features):
            print(f"Warning: Using {len(available_features)}/{len(features)} features")
        features = available_features

        df[features] = feature_scaler.transform(df[features])

        if use_log_prices:
            df['close'] = np.log(df['close'])
        df['close'] = price_scaler.transform(df['close'].values.reshape(-1, 1)).flatten()

        from common.data_utils import create_sequences
        X_test, y_test, current_prices = create_sequences(df, features, lookback, horizon)
        print(f"Created {len(X_test):,} sequences")
        return X_test, y_test, current_prices, features

    raise ValueError(
        f"Unrecognised test_path: {test_path}\n"
        "Expected: prepared-data directory, *.npy (2D), or *.csv"
    )


def test_model(model_dir=None, test_path=None, batch_size=1024):
    # ------------------------------------------------------------------ #
    # Auto-select latest model when model_dir not given
    # ------------------------------------------------------------------ #
    if model_dir is None:
        base_dir = 'lstm_models'
        if not os.path.exists(base_dir):
            raise FileNotFoundError(
                "No lstm_models folder found.\n"
                "Train a model first: python lstm_train_price.py"
            )
        run_dirs = [d for d in os.listdir(base_dir) if d.startswith('L')]
        if not run_dirs:
            raise FileNotFoundError(
                "No trained models found.\n"
                "Train a model first: python lstm_train_price.py"
            )
        model_dir = os.path.join(base_dir, sorted(run_dirs)[-1])
        print(f"Using latest model: {model_dir}")

    if os.path.sep not in model_dir and not os.path.exists(model_dir):
        model_dir = os.path.join('lstm_models', model_dir)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_id = extract_model_id(model_dir)
    config   = joblib.load(os.path.join(model_dir, 'model_config.pkl'))
    data_id  = extract_data_id(config)

    # ------------------------------------------------------------------ #
    # Auto-detect test data when test_path not given
    # Strategy 1: same prepared-data folder the model was trained on
    # Strategy 2: search lstm_prepared_data by data_id
    # Strategy 3: try train CSV basename  train→test replacement
    # Strategy 4: newest folder in lstm_prepared_data
    # ------------------------------------------------------------------ #
    if test_path is None:
        train_path = config.get('train_path', '')

        # Strategy 1 — trained on prepared directory → use same dir
        if train_path and os.path.isdir(train_path):
            x3d = os.path.join(train_path, 'X_test_3d.npy')
            if os.path.exists(x3d):
                test_path = train_path
                print(f"✓ Auto-detected: model's prepared directory  ({test_path})")

        # Strategy 2 — search lstm_prepared_data by data_id prefix
        if test_path is None and data_id not in ('UNKNOWN', ''):
            prep_base = 'lstm_prepared_data'
            if os.path.exists(prep_base):
                candidates = sorted([
                    d for d in os.listdir(prep_base)
                    if d.startswith(data_id)
                    and os.path.isdir(os.path.join(prep_base, d))
                    and os.path.exists(os.path.join(prep_base, d, 'X_test_3d.npy'))
                ])
                if candidates:
                    test_path = os.path.join(prep_base, candidates[-1])
                    print(f"✓ Auto-detected by data_id ({data_id}):  {test_path}")

        # Strategy 3 — trained on CSV → look for sibling test CSV
        if test_path is None and train_path and train_path.endswith('.csv'):
            test_csv = os.path.join(
                os.path.dirname(train_path) or '.',
                os.path.basename(train_path).replace('train', 'test'),
            )
            if os.path.exists(test_csv):
                test_path = test_csv
                print(f"✓ Auto-detected test CSV: {test_path}")

        # Strategy 4 — newest prepared directory with 3D sequences
        if test_path is None:
            prep_base = 'lstm_prepared_data'
            if os.path.exists(prep_base):
                candidates = sorted([
                    d for d in os.listdir(prep_base)
                    if '_prep_' in d
                    and os.path.isdir(os.path.join(prep_base, d))
                    and os.path.exists(os.path.join(prep_base, d, 'X_test_3d.npy'))
                ])
                if candidates:
                    test_path = os.path.join(prep_base, candidates[-1])
                    print(f"✓ Auto-detected latest prepared data:  {test_path}")

        if test_path is None:
            raise FileNotFoundError(
                f"Cannot auto-detect test data.\n"
                f"Model trained on: {train_path or 'UNKNOWN'}\n"
                f"Data ID:          {data_id}\n\n"
                "Provide explicitly:\n"
                "  --test_path lstm_prepared_data/E101_prep_eth_universal_full_1m\n"
                "  --test_path data/test.csv\n"
                "  --test_path data/test_2d.npy"
            )

    print(f"Test data: {test_path}")

    # ------------------------------------------------------------------ #
    # Create output directory
    # ------------------------------------------------------------------ #
    test_base = 'lstm_data'
    os.makedirs(test_base, exist_ok=True)

    test_id  = get_next_test_id(test_base)
    date_str = datetime.now().strftime('%d%m')
    run_name = f"{test_id}_test_results_{model_id}_{data_id}_{date_str}"
    run_dir  = os.path.join(test_base, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Test ID:  {test_id}")
    print(f"Model ID: {model_id}")
    print(f"Data ID:  {data_id}")
    print(f"Results:  {run_dir}")

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    print("\nLoading model...")
    feature_scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
    price_scaler   = joblib.load(os.path.join(model_dir, 'price_scaler.pkl'))

    features     = config['features']
    lookback     = config['lookback']
    horizon      = config['horizon']
    hidden_size  = config['hidden_size']
    num_layers   = config['num_layers']
    dropout      = config['dropout']
    feature_set  = config.get('feature_set')
    use_log_prices = _read_use_log_prices(config)

    print(f"Config:     lookback={lookback}, horizon={horizon}")
    print(f"Features:   {len(features)}")
    if feature_set:
        print(f"Feature set: {feature_set}")
    print(f"Log prices: {use_log_prices}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:     {device}")

    from lstm_train_price import LSTMPriceModel
    model = LSTMPriceModel(len(features), hidden_size, num_layers, dropout).to(device)
    model.load_state_dict(torch.load(
        os.path.join(model_dir, 'lstm_price_model.pth'), map_location=device))
    model.eval()
    print("Model loaded ✓")

    # ------------------------------------------------------------------ #
    # Load test data
    # ------------------------------------------------------------------ #
    print(f"\nLoading test data: {test_path}")
    X_test, y_test, current_prices, features = load_test_data(
        test_path, config, lookback, horizon,
        features, feature_scaler, price_scaler, use_log_prices,
    )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    print(f"\nGenerating predictions (batch_size={batch_size})...")
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, len(X_test), batch_size), desc="Predicting"):
            X_batch = torch.FloatTensor(X_test[i:i + batch_size]).to(device)
            out = model(X_batch).squeeze().cpu().numpy()
            predictions.extend([out.item()] if out.ndim == 0 else out.tolist())

    predictions = np.array(predictions)
    print(f"Generated {len(predictions):,} predictions")

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)

    metrics, y_true_price, y_pred_price = calculate_metrics(
        y_test, predictions, price_scaler, use_log_prices)

    if not np.all(current_prices == 0):
        if use_log_prices:
            cur_log  = price_scaler.inverse_transform(
                current_prices.reshape(-1, 1)).flatten()
            current_real = np.exp(cur_log)
        else:
            current_real = price_scaler.inverse_transform(
                current_prices.reshape(-1, 1)).flatten()
    else:
        current_real = np.zeros(len(y_true_price))

    dir_acc = calculate_directional_accuracy(y_true_price, y_pred_price, current_real)

    print_metrics_summary(metrics, dir_acc if dir_acc > 0 else None)
    interpret_metrics(metrics, dir_acc if dir_acc > 0 else None)

    plot_test_results(y_true_price, y_pred_price, current_real, run_dir, horizon)

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    print("\nSaving results...")

    actual_dir = (np.sign(y_true_price - current_real)
                  if not np.all(current_real == 0) else np.zeros(len(y_true_price)))
    pred_dir   = (np.sign(y_pred_price - current_real)
                  if not np.all(current_real == 0) else np.zeros(len(y_pred_price)))

    pd.DataFrame({
        'actual':     y_true_price,
        'predicted':  y_pred_price,
        'current':    current_real,
        'error_$':    y_pred_price - y_true_price,
        'error_%':    ((y_pred_price - y_true_price) / y_true_price) * 100,
        'actual_dir': actual_dir,
        'pred_dir':   pred_dir,
        'correct_dir': (actual_dir == pred_dir).astype(int),
    }).to_csv(os.path.join(run_dir, 'results.csv'), index=False)
    print("  ✓ results.csv")

    with open(os.path.join(run_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("LSTM PRICE PREDICTION TEST SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test ID:    {test_id}\n")
        f.write(f"Model ID:   {model_id}\n")
        f.write(f"Data ID:    {data_id}\n")
        f.write(f"Date:       {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
        f.write(f"Horizon:    t+{horizon}\n")
        f.write(f"Samples:    {len(y_test):,}\n\n")
        f.write("METRICS:\n")
        f.write(f"  MAE:      ${metrics['mae']:.2f}\n")
        f.write(f"  RMSE:     ${metrics['rmse']:.2f}\n")
        f.write(f"  MAPE:     {metrics['mape']:.2f}%\n")
        f.write(f"  R²:       {metrics['r2']:.4f}\n")
        f.write(f"  Dir Acc:  {dir_acc:.2f}%\n")
    print("  ✓ summary.txt")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Test ID:  {test_id}")
    print(f"Model ID: {model_id}")
    print(f"Data ID:  {data_id}")
    print(f"MAE:      ${metrics['mae']:,.2f}")
    print(f"MAPE:     {metrics['mape']:.2f}%")
    print(f"R²:       {metrics['r2']:.4f}")
    print(f"Dir:      {dir_acc:.2f}%")
    print(f"Output:   {run_dir}")
    print("=" * 70)

    metrics['dir_acc'] = dir_acc
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1024)

    args = parser.parse_args()

    try:
        test_model(
            model_dir=args.model_dir,
            test_path=args.test_path,
            batch_size=args.batch_size,
        )
    except FileNotFoundError as e:
        print("\n" + "=" * 70)
        print("ERROR")
        print("=" * 70)
        print(str(e))
        print("=" * 70)
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR")
        print("=" * 70)
        print(str(e))
        import traceback
        traceback.print_exc()
        print("=" * 70)
        raise
