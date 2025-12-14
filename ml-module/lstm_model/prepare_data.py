"""
Data preparation for LSTM training
"""
import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import warnings

from feature_engineering import prepare_features, select_features

warnings.filterwarnings('ignore')


SYMBOL_ID_MAP = {
    'btc': 'B', 'eth': 'E', 'bnb': 'N', 'sol': 'S', 'ada': 'A',
    'xrp': 'X', 'doge': 'D', 'dot': 'P', 'matic': 'M', 'avax': 'V'
}

INTERVAL_MAP = {
    60: '1m', 300: '5m', 900: '15m', 1800: '30m',
    3600: '1h', 14400: '4h', 86400: '1d'
}


def detect_symbol(filename):
    """Detect cryptocurrency symbol from filename"""
    filename_lower = filename.lower()
    for symbol in SYMBOL_ID_MAP.keys():
        if symbol in filename_lower:
            return symbol
    return 'crypto'


def get_next_id(output_dir, symbol):
    """Get next available ID for symbol"""
    prefix = SYMBOL_ID_MAP.get(symbol, 'C')
    existing = [d for d in os.listdir(output_dir) if d.startswith(prefix) and '_prep_' in d]

    if not existing:
        return f"{prefix}001"

    numbers = []
    for d in existing:
        try:
            num = int(d[1:4])
            numbers.append(num)
        except:
            continue

    next_num = max(numbers) + 1 if numbers else 1
    return f"{prefix}{next_num:03d}"


def detect_interval(df):
    df_temp = df.copy()
    try:
        df_temp['datetime'] = pd.to_datetime(df_temp['open_time'], unit='ms', errors='coerce')
        if df_temp['datetime'].isna().all():
            raise ValueError("Not Unix timestamps")
    except (ValueError, TypeError):
        df_temp['datetime'] = pd.to_datetime(df_temp['open_time'])

    time_diffs = df_temp['datetime'].diff().dropna()
    if len(time_diffs) == 0:
        return 'unknown'

    median_diff = time_diffs.median().total_seconds()

    return INTERVAL_MAP.get(int(median_diff), f"{int(median_diff)}s")


def get_scaler(scaler_type):
    """Get scaler instance by name"""
    scalers = {
        'minmax': MinMaxScaler(),
        'standard': StandardScaler(),
        'robust': RobustScaler()
    }
    return scalers.get(scaler_type.lower())


def create_sequences(data, features, lookback, horizon):
    """Create LSTM sequences (3D)"""
    data_array = data[features].values.astype(np.float32)
    prices = data['close'].values.astype(np.float32)

    n_samples = len(data) - lookback - horizon + 1
    if n_samples <= 0:
        return None, None

    X = np.zeros((n_samples, lookback, len(features)), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        X[i] = data_array[i:i + lookback]
        y[i] = prices[i + lookback + horizon - 1]

    return X, y


def split_by_months(df, train_months, agent_months, test_months):
    if 'open_time' not in df.columns:
        raise ValueError("Missing 'open_time' column")

    df_copy = df.copy()

    try:
        df_copy['datetime'] = pd.to_datetime(df_copy['open_time'], unit='ms', errors='coerce')
        if df_copy['datetime'].isna().all():
            raise ValueError("Not Unix timestamps")
    except (ValueError, TypeError):
        df_copy['datetime'] = pd.to_datetime(df_copy['open_time'])

    df_copy = df_copy.sort_values('datetime').reset_index(drop=True)

    latest_date = df_copy['datetime'].max()
    earliest_date = df_copy['datetime'].min()

    test_start = latest_date - pd.Timedelta(days=test_months * 30)

    if test_start < earliest_date:
        test_start = earliest_date

    test_mask = df_copy['datetime'] > test_start
    test_df = df_copy[test_mask].copy()

    if agent_months > 0:
        agent_start = test_start - pd.Timedelta(days=agent_months * 30)
        if agent_start < earliest_date:
            agent_start = earliest_date
        agent_mask = (df_copy['datetime'] > agent_start) & (df_copy['datetime'] <= test_start)
        agent_df = df_copy[agent_mask].copy()
    else:
        agent_df = None

    train_end = agent_start if agent_months > 0 else test_start
    train_start = train_end - pd.Timedelta(days=train_months * 30)

    if train_start < earliest_date:
        train_start = earliest_date

    train_mask = (df_copy['datetime'] > train_start) & (df_copy['datetime'] <= train_end)
    train_df = df_copy[train_mask].copy()

    date_ranges = {
        'train': (train_df['datetime'].min(), train_df['datetime'].max()) if len(train_df) > 0 else (None, None),
        'agent': (agent_df['datetime'].min(), agent_df['datetime'].max()) if agent_df is not None and len(agent_df) > 0 else None,
        'test': (test_df['datetime'].min(), test_df['datetime'].max()) if len(test_df) > 0 else (None, None)
    }

    train_df = train_df.drop('datetime', axis=1)
    test_df = test_df.drop('datetime', axis=1)
    if agent_df is not None:
        agent_df = agent_df.drop('datetime', axis=1)

    return train_df, agent_df, test_df, date_ranges


def split_by_percentage(df, train_pct, agent_pct, test_pct):
    if 'open_time' not in df.columns:
        raise ValueError("Missing 'open_time' column")

    total_pct = train_pct + agent_pct + test_pct
    if abs(total_pct - 100.0) > 0.01:
        raise ValueError(f"Percentages must sum to 100, got {total_pct}")

    df_copy = df.copy()

    try:
        df_copy['datetime'] = pd.to_datetime(df_copy['open_time'], unit='ms', errors='coerce')
        if df_copy['datetime'].isna().all():
            raise ValueError("Not Unix timestamps")
    except (ValueError, TypeError):
        df_copy['datetime'] = pd.to_datetime(df_copy['open_time'])

    df_copy = df_copy.sort_values('datetime').reset_index(drop=True)

    n = len(df_copy)
    train_end_idx = int(n * train_pct / 100)
    agent_end_idx = train_end_idx + int(n * agent_pct / 100)

    train_df = df_copy.iloc[:train_end_idx].copy()

    if agent_pct > 0:
        agent_df = df_copy.iloc[train_end_idx:agent_end_idx].copy()
    else:
        agent_df = None

    test_df = df_copy.iloc[agent_end_idx:].copy()

    date_ranges = {
        'train': (train_df['datetime'].min(), train_df['datetime'].max()) if len(train_df) > 0 else (None, None),
        'agent': (agent_df['datetime'].min(), agent_df['datetime'].max()) if agent_df is not None and len(agent_df) > 0 else None,
        'test': (test_df['datetime'].min(), test_df['datetime'].max()) if len(test_df) > 0 else (None, None)
    }

    train_df = train_df.drop('datetime', axis=1)
    test_df = test_df.drop('datetime', axis=1)
    if agent_df is not None:
        agent_df = agent_df.drop('datetime', axis=1)

    return train_df, agent_df, test_df, date_ranges


def prepare_data(
    data_path,
    output_dir='lstm_prepared_data',
    feature_set='full',
    train_months=None,
    agent_months=None,
    test_months=None,
    train_pct=70.0,
    agent_pct=0.0,
    test_pct=30.0,
    lookback=48,
    horizon=1,
    scaler_features='standard',
    scaler_price='standard',
    use_log_prices=True
):
    """Prepare data for LSTM training"""

    print("=" * 70)
    print("DATA PREPARATION FOR LSTM")
    print("=" * 70)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Detect symbol and interval
    symbol = detect_symbol(os.path.basename(data_path))
    print(f"\nDetected symbol: {symbol.upper()}")

    # Load data
    print(f"\n1. Loading data: {data_path}")
    try:
        if data_path.endswith('.npy'):
            data = np.load(data_path, allow_pickle=True)
            df = pd.DataFrame(data)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format. Use .csv or .npy")
    except Exception as e:
        raise ValueError(f"Failed to load data: {e}")

    print(f"   Loaded {len(df)} records")
    print(f"   Columns: {list(df.columns)}")

    min_required = 72
    if len(df) < min_required:
        raise ValueError(
            f"Insufficient input data: {len(df)} records. "
            f"Need at least {min_required} records for feature generation. "
            f"Technical indicators require historical data."
        )

    # Get raw data date range
    try:
        df_temp = df.copy()
        df_temp['datetime'] = pd.to_datetime(df_temp['open_time'], unit='ms')
    except:
        df_temp['datetime'] = pd.to_datetime(df_temp['open_time'])

    raw_date_start = df_temp['datetime'].min()
    raw_date_end = df_temp['datetime'].max()
    print(f"   Raw data range: {raw_date_start} to {raw_date_end}")

    # Detect interval
    interval = detect_interval(df)
    print(f"   Detected interval: {interval}")

    # Verify and clean columns
    print("\n2. Verifying and cleaning data")
    required = ['open_time', 'open', 'high', 'low', 'close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keep_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    if 'close_time' in df.columns:
        keep_cols.insert(1, 'close_time')
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    print(f"   Kept {len(keep_cols)} columns: {keep_cols}")

    print(f"\n3. Generating features (set: {feature_set})")
    df = prepare_features(df)
    features = select_features(df, feature_set=feature_set, validate=False)
    features = [f for f in features if f != 'close']

    print(f"   Generated {len(features)} features")
    if len(features) == 0:
        raise ValueError(f"No features generated for set: {feature_set}")

    print("\n4. Removing NaN values")
    initial_len = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    nan_count_before = df.isna().sum().sum()
    print(f"   Total NaN values before dropna: {nan_count_before}")

    if nan_count_before > 0:
        nan_by_column = df.isna().sum()
        print(f"   NaN by column (top 5):")
        for col, count in nan_by_column.nlargest(5).items():
            print(f"     {col}: {count}")

    df = df.dropna()
    removed = initial_len - len(df)
    print(f"   Removed {removed} rows ({removed/initial_len*100:.1f}%)")
    print(f"   Remaining: {len(df)} records")

    if len(df) < lookback + horizon + 100:
        raise ValueError(
            f"Insufficient data after cleaning: {len(df)} rows. "
            f"Need at least {lookback + horizon + 100} rows. "
            f"Try using a smaller lookback value or more input data."
        )

    print(f"\n5. Splitting data")

    use_months = train_months is not None and test_months is not None

    if use_months:
        print(f"   Mode: BY MONTHS")
        print(f"   Train: {train_months} months")
        if agent_months and agent_months > 0:
            print(f"   Agent: {agent_months} months")
        print(f"   Test:  {test_months} months")

        train_df, agent_df, test_df, date_ranges = split_by_months(
            df, train_months, agent_months or 0, test_months
        )
    else:
        print(f"   Mode: BY PERCENTAGE")
        print(f"   Train: {train_pct}%")
        if agent_pct > 0:
            print(f"   Agent: {agent_pct}%")
        print(f"   Test:  {test_pct}%")

        train_df, agent_df, test_df, date_ranges = split_by_percentage(
            df, train_pct, agent_pct, test_pct
        )

    print(f"   Train: {len(train_df)} records")
    if date_ranges['train'][0]:
        print(f"   Train dates: {date_ranges['train'][0]} to {date_ranges['train'][1]}")
    if agent_df is not None and len(agent_df) > 0:
        print(f"   Agent: {len(agent_df)} records")
        if date_ranges['agent']:
            print(f"   Agent dates: {date_ranges['agent'][0]} to {date_ranges['agent'][1]}")
    print(f"   Test:  {len(test_df)} records")
    if date_ranges['test'][0]:
        print(f"   Test dates: {date_ranges['test'][0]} to {date_ranges['test'][1]}")

    if len(train_df) < lookback + horizon:
        raise ValueError(f"Insufficient training data: {len(train_df)} rows")
    if len(test_df) < lookback + horizon:
        raise ValueError(f"Insufficient test data: {len(test_df)} rows")

    # Create scalers
    print(f"\n6. Creating scalers")
    print(f"   Features: {scaler_features}")
    print(f"   Price:    {scaler_price}")
    print(f"   Log prices: {use_log_prices}")

    feature_scaler = get_scaler(scaler_features)
    price_scaler = get_scaler(scaler_price)

    # Fit scalers on train only
    print("   Fitting on train data...")
    feature_scaler.fit(train_df[features])

    if use_log_prices:
        train_df['close'] = np.log(train_df['close'])

    price_scaler.fit(train_df['close'].values.reshape(-1, 1))

    # Transform all splits
    print("\n7. Transforming data")
    train_df[features] = feature_scaler.transform(train_df[features])
    test_df[features] = feature_scaler.transform(test_df[features])

    if use_log_prices:
        test_df['close'] = np.log(test_df['close'])

    train_df['close'] = price_scaler.transform(train_df['close'].values.reshape(-1, 1)).flatten()
    test_df['close'] = price_scaler.transform(test_df['close'].values.reshape(-1, 1)).flatten()

    if agent_df is not None:
        agent_df[features] = feature_scaler.transform(agent_df[features])
        if use_log_prices:
            agent_df['close'] = np.log(agent_df['close'])
        agent_df['close'] = price_scaler.transform(agent_df['close'].values.reshape(-1, 1)).flatten()

    print("   Transform complete")

    # Create sequences (3D)
    print(f"\n8. Creating LSTM sequences")
    print(f"   Lookback: {lookback}")
    print(f"   Horizon:  {horizon}")

    X_train, y_train = create_sequences(train_df, features, lookback, horizon)
    X_test_3d, y_test_3d = create_sequences(test_df, features, lookback, horizon)

    if X_train is None:
        raise ValueError("Insufficient data for train sequences")

    print(f"   Train: X={X_train.shape}, y={y_train.shape}")
    print(f"   Test (3D):  X={X_test_3d.shape}, y={y_test_3d.shape}")

    # Create 2D test data (without sequences)
    print("\n9. Creating 2D test data")
    test_2d = test_df[features + ['close']].values.astype(np.float32)
    print(f"   Test (2D): {test_2d.shape}")

    # Create output directory with new naming
    print("\n10. Saving prepared data")
    os.makedirs(output_dir, exist_ok=True)

    data_id = get_next_id(output_dir, symbol)
    dir_name = f"{data_id}_prep_{symbol}_data_{feature_set}_{interval}"
    full_path = os.path.join(output_dir, dir_name)
    os.makedirs(full_path, exist_ok=True)

    print(f"   Data ID: {data_id}")
    print(f"   Output: {full_path}")

    # Save arrays
    np.save(os.path.join(full_path, 'X_train.npy'), X_train)
    np.save(os.path.join(full_path, 'y_train.npy'), y_train)
    np.save(os.path.join(full_path, 'X_test_3d.npy'), X_test_3d)
    np.save(os.path.join(full_path, 'y_test_3d.npy'), y_test_3d)
    np.save(os.path.join(full_path, 'test_2d.npy'), test_2d)

    # Save raw data for agent (no sequences)
    if agent_df is not None:
        agent_data = agent_df[features + ['close']].values
        np.save(os.path.join(full_path, 'agent_data.npy'), agent_data)
        print(f"   Saved agent_data.npy: {agent_data.shape}")

    # Save scalers
    joblib.dump(feature_scaler, os.path.join(full_path, 'scaler_features.pkl'))
    joblib.dump(price_scaler, os.path.join(full_path, 'scaler_price.pkl'))

    # Save metadata
    metadata = {
        'data_id': data_id,
        'source_file': os.path.basename(data_path),
        'symbol': symbol,
        'interval': interval,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'feature_set': feature_set,
        'features': features,
        'n_features': len(features),
        'split_mode': 'months' if use_months else 'percentage',
        'train_months': train_months if use_months else None,
        'agent_months': agent_months if use_months else None,
        'test_months': test_months if use_months else None,
        'train_pct': train_pct if not use_months else None,
        'agent_pct': agent_pct if not use_months else None,
        'test_pct': test_pct if not use_months else None,
        'lookback': lookback,
        'horizon': horizon,
        'scaler_features': scaler_features,
        'scaler_price': scaler_price,
        'use_log_prices': use_log_prices,
        'raw_data_range': {
            'start': str(raw_date_start),
            'end': str(raw_date_end)
        },
        'train_date_range': {
            'start': str(date_ranges['train'][0]) if date_ranges['train'][0] else None,
            'end': str(date_ranges['train'][1]) if date_ranges['train'][1] else None
        },
        'test_date_range': {
            'start': str(date_ranges['test'][0]) if date_ranges['test'][0] else None,
            'end': str(date_ranges['test'][1]) if date_ranges['test'][1] else None
        },
        'data_shapes': {
            'X_train': list(X_train.shape),
            'y_train': list(y_train.shape),
            'X_test_3d': list(X_test_3d.shape),
            'y_test_3d': list(y_test_3d.shape),
            'test_2d': list(test_2d.shape)
        },
        'data_counts': {
            'train': len(train_df),
            'agent': len(agent_df) if agent_df is not None else 0,
            'test': len(test_df)
        }
    }

    if agent_df is not None and date_ranges['agent']:
        metadata['agent_date_range'] = {
            'start': str(date_ranges['agent'][0]) if date_ranges['agent'][0] else None,
            'end': str(date_ranges['agent'][1]) if date_ranges['agent'][1] else None
        }

    with open(os.path.join(full_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save improved log
    log_path = os.path.join(full_path, 'preparation.log')
    with open(log_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DATA PREPARATION LOG\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Data ID:  {data_id}\n")
        f.write(f"Source:   {data_path}\n")
        f.write(f"Symbol:   {symbol.upper()}\n")
        f.write(f"Interval: {interval}\n")
        f.write(f"Date:     {metadata['timestamp']}\n\n")

        f.write("RAW DATA:\n")
        f.write(f"  Date range: {raw_date_start} to {raw_date_end}\n")
        f.write(f"  Records:    {initial_len}\n\n")

        f.write("TRAIN DATA:\n")
        f.write(f"  File:       X_train.npy, y_train.npy\n")
        f.write(f"  Date range: {date_ranges['train'][0]} to {date_ranges['train'][1]}\n")
        f.write(f"  Records:    {len(train_df)}\n")
        f.write(f"  Sequences:  {X_train.shape[0]}\n")
        f.write(f"  Shape:      X={X_train.shape}, y={y_train.shape}\n\n")

        if agent_df is not None:
            f.write("AGENT DATA:\n")
            f.write(f"  File:       agent_data.npy\n")
            f.write(f"  Date range: {date_ranges['agent'][0]} to {date_ranges['agent'][1]}\n")
            f.write(f"  Records:    {len(agent_df)}\n")
            f.write(f"  Shape:      {agent_data.shape}\n\n")

        f.write("TEST DATA:\n")
        f.write(f"  File (3D):  X_test_3d.npy, y_test_3d.npy\n")
        f.write(f"  File (2D):  test_2d.npy\n")
        f.write(f"  Date range: {date_ranges['test'][0]} to {date_ranges['test'][1]}\n")
        f.write(f"  Records:    {len(test_df)}\n")
        f.write(f"  Sequences:  {X_test_3d.shape[0]}\n")
        f.write(f"  Shape (3D): X={X_test_3d.shape}, y={y_test_3d.shape}\n")
        f.write(f"  Shape (2D): {test_2d.shape}\n\n")

        f.write("CONFIGURATION:\n")
        f.write(f"  Feature set: {feature_set}\n")
        f.write(f"  Features:    {len(features)}\n")
        f.write(f"  Lookback:    {lookback}\n")
        f.write(f"  Horizon:     {horizon}\n")
        f.write(f"  Train months: {train_months}\n")
        f.write(f"  Agent months: {agent_months}\n")
        f.write(f"  Test months:  {test_months}\n\n")

        f.write("SCALERS:\n")
        f.write(f"  Features:   {scaler_features}\n")
        f.write(f"  Price:      {scaler_price}\n")
        f.write(f"  Log prices: {use_log_prices}\n")

    print(f"   Saved: X_train.npy, y_train.npy")
    print(f"   Saved: X_test_3d.npy, y_test_3d.npy")
    print(f"   Saved: test_2d.npy")
    print(f"   Saved: scaler_features.pkl, scaler_price.pkl")
    print(f"   Saved: metadata.json, preparation.log")

    print("\n" + "=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Data ID: {data_id}")
    print(f"Symbol:  {symbol.upper()}")
    print(f"Output:  {full_path}")
    print(f"Train sequences: {X_train.shape[0]}")
    print(f"Test sequences (3D): {X_test_3d.shape[0]}")
    print(f"Test records (2D): {test_2d.shape[0]}")
    print("=" * 70)

    return full_path, data_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--output_dir', type=str, default='lstm_prepared_data')
    parser.add_argument('--feature_set', type=str, default='full',
                       choices=['minimal', 'technical', 'full', 'returns_only',
                               'no_time', 'momentum', 'volatility_focused'])
    parser.add_argument('--train_months', type=int, default=3)
    parser.add_argument('--agent_months', type=int, default=0)
    parser.add_argument('--test_months', type=int, default=1)
    parser.add_argument('--lookback', type=int, default=48)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--scaler_features', type=str, default='standard',
                       choices=['minmax', 'standard', 'robust'])
    parser.add_argument('--scaler_price', type=str, default='standard',
                       choices=['minmax', 'standard', 'robust'])
    parser.add_argument('--no_log_prices', action='store_true')

    args = parser.parse_args()

    try:
        prepare_data(
            data_path=args.data_path,
            output_dir=args.output_dir,
            feature_set=args.feature_set,
            train_months=args.train_months,
            agent_months=args.agent_months,
            test_months=args.test_months,
            lookback=args.lookback,
            horizon=args.horizon,
            scaler_features=args.scaler_features,
            scaler_price=args.scaler_price,
            use_log_prices=not args.no_log_prices
        )
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR")
        print("=" * 70)
        print(f"{type(e).__name__}: {e}")
        print("=" * 70)
        import sys
        sys.exit(1)