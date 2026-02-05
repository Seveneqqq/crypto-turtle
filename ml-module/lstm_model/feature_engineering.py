"""
Feature engineering module for cryptocurrency OHLCV data
"""
import pandas as pd
import numpy as np


def prepare_features(df):
    """
    Generate technical features from raw OHLCV data

    Args:
        df: DataFrame with columns: open_time, open, high, low, close, volume

    Returns:
        DataFrame with additional feature columns
    """
    df = df.copy()

    # Log-returns
    for window in [1, 5, 15, 60]:
        df[f'return_{window}'] = np.log(df['close'] / df['close'].shift(window))

    # SMA and ratios
    for window in [7, 14, 30]:
        sma = df['close'].rolling(window).mean()
        df[f'close_sma_{window}'] = df['close'] / sma

    # EMA
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()

    # MACD
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Volatility
    for window in [7, 14, 30]:
        df[f'volatility_{window}'] = df['return_1'].rolling(window).std()

    # Volume
    volume_sma = df['volume'].rolling(20).mean()
    df['relative_volume'] = df['volume'] / volume_sma

    # Bollinger Bands
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

    # High-Low range
    df['hl_range'] = (df['high'] - df['low']) / df['close']

    # Time features (cyclical encoding)
    dt = pd.to_datetime(df['open_time'])
    hour = dt.dt.hour
    dow = dt.dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    return df


def get_feature_columns():
    """Returns list of feature column names (excluding base and helper columns)"""
    features = []

    # Returns
    features.extend([f'return_{w}' for w in [1, 5, 15, 60]])

    # Technical indicators
    features.extend([f'close_sma_{w}' for w in [7, 14, 30]])
    features.extend(['macd', 'macd_signal', 'macd_diff', 'rsi'])

    # Volatility and volume
    features.extend([f'volatility_{w}' for w in [7, 14, 30]])
    features.append('relative_volume')

    # Price patterns
    features.extend(['bb_position', 'hl_range'])

    # Time features
    features.extend(['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'])

    return features


def get_feature_sets():
    """
    Returns predefined feature sets

    Returns:
        dict: Feature set name -> list of feature names
    """
    all_features = get_feature_columns()

    return {
        'minimal': [
            'return_1', 'return_5',
            'close_sma_14',
            'rsi',
            'volatility_14',
            'relative_volume'
        ],
        'technical': [
            'return_1', 'return_5', 'return_15',
            'close_sma_7', 'close_sma_14', 'close_sma_30',
            'macd', 'macd_signal', 'macd_diff',
            'rsi',
            'volatility_7', 'volatility_14',
            'bb_position',
            'hl_range'
        ],
        'full': all_features,
        'returns_only': [f'return_{w}' for w in [1, 5, 15, 60]],
        'no_time': [f for f in all_features if not any(x in f for x in ['hour', 'dow'])],
        'momentum': [
            'return_1', 'return_5', 'return_15', 'return_60',
            'macd', 'macd_signal', 'macd_diff',
            'rsi',
            'close_sma_7', 'close_sma_14', 'close_sma_30'
        ],
        'volatility_focused': [
            'return_1', 'return_5',
            'volatility_7', 'volatility_14', 'volatility_30',
            'bb_position',
            'hl_range',
            'relative_volume'
        ]
    }


def get_binance_columns():
    """Returns list of standard Binance API column names to exclude"""
    return [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
        'ignore',
        # Alternative naming
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume',
        'Ignore'
    ]


def validate_features(df, required_features):
    """
    Validate that DataFrame contains all required features

    Args:
        df: DataFrame to check
        required_features: List of required feature names

    Returns:
        tuple: (is_valid, missing_features)
    """
    missing = [f for f in required_features if f not in df.columns]
    return len(missing) == 0, missing


def select_features(df, feature_set=None, custom_features=None, validate=True):
    """
    Select features from DataFrame

    Args:
        df: DataFrame with data
        feature_set: Name of predefined set ('minimal', 'technical', 'full', etc.)
        custom_features: List of custom feature names (overrides feature_set)
        validate: If True, validate features exist; if False, use all non-base columns

    Returns:
        list: Feature column names
    """
    # Custom features override everything
    if custom_features is not None:
        if validate:
            is_valid, missing = validate_features(df, custom_features)
            if not is_valid:
                raise ValueError(f"Missing features: {missing}")
            return custom_features
        else:
            available = [f for f in custom_features if f in df.columns]
            return available

    # Feature set
    if feature_set is not None:
        feature_sets = get_feature_sets()
        if feature_set not in feature_sets:
            raise ValueError(f"Unknown feature set: {feature_set}. Available: {list(feature_sets.keys())}")

        features = feature_sets[feature_set]

        if validate:
            is_valid, missing = validate_features(df, features)
            if not is_valid:
                raise ValueError(f"Missing features from set '{feature_set}': {missing}")
        else:
            features = [f for f in features if f in df.columns]

        return features

    # No validation - use all non-base columns
    if not validate:
        base_cols = get_binance_columns()
        base_cols_lower = {c.lower().replace('_', '').replace(' ', '') for c in base_cols}

        features = []
        for col in df.columns:
            col_clean = col.lower().replace('_', '').replace(' ', '')
            if col_clean not in base_cols_lower:
                features.append(col)

        return features

    # Default: use all standard features with validation
    features = get_feature_columns()
    is_valid, missing = validate_features(df, features)

    if not is_valid:
        raise ValueError(f"Missing standard features: {missing}")

    return features