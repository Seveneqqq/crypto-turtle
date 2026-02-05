"""
Common data utilities for LSTM pipeline
"""
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


def load_data(data_path, use_subset=False, subset_size=100000):
    """Load data from prepared directory or CSV file"""
    if not os.path.sep in data_path and not os.path.exists(data_path):
        if not data_path.endswith('.csv'):
            candidate = os.path.join('lstm_prepared_data', data_path)
            if os.path.exists(candidate):
                data_path = candidate

    if os.path.isdir(data_path):
        metadata_path = os.path.join(data_path, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise ValueError(f"Directory missing metadata.json: {data_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        X_train = np.load(os.path.join(data_path, 'X_train.npy'))
        y_train = np.load(os.path.join(data_path, 'y_train.npy'))

        X_val_path = os.path.join(data_path, 'X_test_3d.npy')
        if os.path.exists(X_val_path):
            X_val = np.load(X_val_path)
            y_val = np.load(os.path.join(data_path, 'y_test_3d.npy'))
        else:
            split_idx = int(len(X_train) * 0.8)
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]

        feature_scaler = joblib.load(os.path.join(data_path, 'scaler_features.pkl'))
        price_scaler = joblib.load(os.path.join(data_path, 'scaler_price.pkl'))

        data = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'feature_scaler': feature_scaler,
            'price_scaler': price_scaler
        }
        return data, True, metadata

    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        if use_subset:
            df = df.tail(subset_size).reset_index(drop=True)
        return df, False, None
    else:
        raise ValueError(f"Invalid data path: {data_path}")


def create_sequences(data, features, lookback=24, horizon=1):
    """Create sequences for price prediction"""
    data_array = data[features].values.astype(np.float32)
    prices = data['close'].values.astype(np.float32)

    n_samples = len(data) - lookback - horizon + 1
    if n_samples <= 0:
        return None, None, None

    X = np.zeros((n_samples, lookback, len(features)), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    current_prices = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        X[i] = data_array[i:i + lookback]
        current_prices[i] = prices[i + lookback - 1]
        y[i] = prices[i + lookback + horizon - 1]

    return X, y, current_prices


def prepare_raw_data(df, features, feature_scaler, price_scaler, use_log_prices,
                     lookback, horizon, val_split=0.2):
    """Prepare raw CSV data for training"""
    from feature_engineering import prepare_features, select_features

    df = prepare_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    feature_set = features.get('set') if isinstance(features, dict) else None
    custom_features = features.get('custom') if isinstance(features, dict) else None
    validate = features.get('validate', True) if isinstance(features, dict) else True

    features_list = select_features(df, feature_set=feature_set,
                                    custom_features=custom_features,
                                    validate=validate)
    features_list = [f for f in features_list if f != 'close']

    if len(features_list) < 5:
        raise ValueError(f"Too few features: {len(features_list)}")

    split_idx = int(len(df) * (1 - val_split))
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    if feature_scaler is None:
        feature_scaler = StandardScaler()
        train_df[features_list] = feature_scaler.fit_transform(train_df[features_list])
    else:
        train_df[features_list] = feature_scaler.transform(train_df[features_list])

    val_df[features_list] = feature_scaler.transform(val_df[features_list])

    if use_log_prices:
        train_df['close'] = np.log(train_df['close'])
        val_df['close'] = np.log(val_df['close'])

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
    X_val, y_val, _ = create_sequences(val_df, features_list, lookback, horizon)

    return X_train, y_train, X_val, y_val, features_list, feature_scaler, price_scaler


def detect_symbol(filename):
    """Detect cryptocurrency symbol from filename"""
    filename_lower = filename.lower()
    symbols = ['btc', 'eth', 'bnb', 'sol', 'ada', 'xrp', 'doge', 'dot', 'matic', 'avax']
    for sym in symbols:
        if sym in filename_lower:
            return sym
    return 'crypto'