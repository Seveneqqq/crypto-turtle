"""
Common data utilities for LSTM pipeline
"""
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


def _resolve_prepared_dir(data_path):
    """
    Resolve a path to a concrete prepared-data directory that contains
    metadata.json + X_train.npy.

    Handles three cases:
      1. data_path is already a concrete prepared dir  (has metadata.json)
      2. data_path is a container folder               (e.g. 'lstm_prepared_data')
         → pick the newest sub-directory that looks like a prepared dir
      3. data_path is a bare name without separators   (e.g. 'E101_prep_eth_...')
         → look inside 'lstm_prepared_data/' for a matching sub-directory

    Returns the resolved absolute-or-relative path, or raises FileNotFoundError.
    """
    # Case 1 — already a concrete prepared dir
    if os.path.isdir(data_path) and os.path.exists(
            os.path.join(data_path, 'metadata.json')):
        return data_path

    # Case 3 — bare name, no path separator → look inside lstm_prepared_data/
    if os.path.sep not in data_path and not os.path.exists(data_path):
        candidate = os.path.join('lstm_prepared_data', data_path)
        if os.path.isdir(candidate) and os.path.exists(
                os.path.join(candidate, 'metadata.json')):
            return candidate

    # Case 2 — container folder: find newest sub-dir with metadata.json
    if os.path.isdir(data_path):
        sub_dirs = sorted([
            d for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
            and os.path.exists(os.path.join(data_path, d, 'metadata.json'))
            and os.path.exists(os.path.join(data_path, d, 'X_train.npy'))
        ])
        if sub_dirs:
            resolved = os.path.join(data_path, sub_dirs[-1])
            print(f"[load_data] Auto-selected prepared dir: {resolved}")
            return resolved

        raise FileNotFoundError(
            f"No valid prepared-data sub-directory found in: {data_path}\n"
            "Each sub-directory must contain metadata.json and X_train.npy.\n"
            "Run data_preparation/prepare_data.py first."
        )

    raise FileNotFoundError(
        f"Prepared-data directory not found: {data_path}"
    )


def load_data(data_path, use_subset=False, subset_size=100000):
    """
    Load data from prepared directory (or container folder) or CSV file.

    Prepared directory layout (universal format from data_preparation/):
        X_train.npy           — 3D train sequences  (n, lookback, features)
        y_train.npy           — train targets        (n,)
        X_test_3d.npy         — 3D test  sequences  (n, lookback, features)
        y_test_3d.npy         — test  targets        (n,)
        scaler_features.pkl   — feature scaler  (or legacy: feature_scaler.pkl)
        scaler_price.pkl      — price scaler    (or legacy: price_scaler.pkl)
        metadata.json

    If data_path is a container folder (e.g. 'lstm_prepared_data') the newest
    valid sub-directory is selected automatically.
    """
    # ------------------------------------------------------------------ #
    # Directory path — resolve to a concrete prepared dir
    # ------------------------------------------------------------------ #
    if os.path.isdir(data_path) or (
            os.path.sep not in data_path
            and not data_path.endswith('.csv')
            and not os.path.exists(data_path)):

        data_path = _resolve_prepared_dir(data_path)

        with open(os.path.join(data_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        # Train sequences
        X_train = np.load(os.path.join(data_path, 'X_train.npy'))
        y_train = np.load(os.path.join(data_path, 'y_train.npy'))

        # Validation sequences (X_test_3d used as val; fallback: split train)
        x_val_path = os.path.join(data_path, 'X_test_3d.npy')
        y_val_path = os.path.join(data_path, 'y_test_3d.npy')

        if os.path.exists(x_val_path) and os.path.exists(y_val_path):
            X_val = np.load(x_val_path)
            y_val = np.load(y_val_path)
        else:
            split_idx = int(len(X_train) * 0.8)
            X_val   = X_train[split_idx:]
            y_val   = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]

        # Scalers — new naming first, legacy as fallback
        def _load_scaler(new_name, legacy_name):
            p = os.path.join(data_path, new_name)
            if not os.path.exists(p):
                p = os.path.join(data_path, legacy_name)
            return joblib.load(p)

        feature_scaler = _load_scaler('scaler_features.pkl', 'feature_scaler.pkl')
        price_scaler   = _load_scaler('scaler_price.pkl',    'price_scaler.pkl')

        data = {
            'X_train': X_train, 'y_train': y_train,
            'X_val':   X_val,   'y_val':   y_val,
            'feature_scaler': feature_scaler,
            'price_scaler':   price_scaler,
        }
        return data, True, metadata

    # ------------------------------------------------------------------ #
    # CSV file path
    # ------------------------------------------------------------------ #
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        if use_subset:
            df = df.tail(subset_size).reset_index(drop=True)
        return df, False, None

    else:
        raise ValueError(
            f"Invalid data path: {data_path}\n"
            "Expected: prepared-data directory, container folder, or .csv file."
        )


def create_sequences(data, features, lookback=24, horizon=1):
    """Create 3-D sequences for LSTM price prediction."""
    data_array    = data[features].values.astype(np.float32)
    prices        = data['close'].values.astype(np.float32)
    n_samples     = len(data) - lookback - horizon + 1

    if n_samples <= 0:
        return None, None, None

    X              = np.zeros((n_samples, lookback, len(features)), dtype=np.float32)
    y              = np.zeros(n_samples, dtype=np.float32)
    current_prices = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        X[i]              = data_array[i:i + lookback]
        current_prices[i] = prices[i + lookback - 1]
        y[i]              = prices[i + lookback + horizon - 1]

    return X, y, current_prices


def prepare_raw_data(df, features, feature_scaler, price_scaler, use_log_prices,
                     lookback, horizon, val_split=0.2):
    """Prepare raw CSV data for training (on-the-fly path)."""
    from feature_engineering import prepare_features, select_features

    df = prepare_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    feature_set     = features.get('set')             if isinstance(features, dict) else None
    custom_features = features.get('custom')          if isinstance(features, dict) else None
    validate        = features.get('validate', True)  if isinstance(features, dict) else True

    has_buy_pressure = 'buy_pressure' in df.columns

    features_list = select_features(
        df,
        feature_set=feature_set,
        custom_features=custom_features,
        validate=validate,
        include_buy_pressure=has_buy_pressure,
    )
    features_list = [f for f in features_list if f != 'close']

    if len(features_list) < 5:
        raise ValueError(f"Too few features: {len(features_list)}")

    split_idx = int(len(df) * (1 - val_split))
    train_df  = df.iloc[:split_idx].copy()
    val_df    = df.iloc[split_idx:].copy()

    if feature_scaler is None:
        feature_scaler = StandardScaler()
        train_df[features_list] = feature_scaler.fit_transform(train_df[features_list])
    else:
        train_df[features_list] = feature_scaler.transform(train_df[features_list])

    val_df[features_list] = feature_scaler.transform(val_df[features_list])

    if use_log_prices:
        train_df['close'] = np.log(train_df['close'])
        val_df['close']   = np.log(val_df['close'])

    if price_scaler is None:
        price_scaler = StandardScaler()
        train_df['close'] = price_scaler.fit_transform(
            train_df['close'].values.reshape(-1, 1)).flatten()
    else:
        train_df['close'] = price_scaler.transform(
            train_df['close'].values.reshape(-1, 1)).flatten()

    val_df['close'] = price_scaler.transform(
        val_df['close'].values.reshape(-1, 1)).flatten()

    X_train, y_train, _ = create_sequences(train_df, features_list, lookback, horizon)
    X_val,   y_val,   _ = create_sequences(val_df,   features_list, lookback, horizon)

    return X_train, y_train, X_val, y_val, features_list, feature_scaler, price_scaler


def detect_symbol(filename):
    """Detect cryptocurrency symbol from filename."""
    filename_lower = filename.lower()
    for sym in ['btc', 'eth', 'bnb', 'sol', 'ada', 'xrp', 'doge', 'dot', 'matic', 'avax']:
        if sym in filename_lower:
            return sym
    return 'crypto'
