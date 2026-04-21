"""
Data utilities for Random Forest model
-- direct port from xgb_model/common/data_utils.py --
All flattening, target scaling and winsorisation logic is identical.
The only difference is the absence of xgboost-specific imports.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler


# =============================================================================
# TARGET SCALER
# =============================================================================

class TargetScaler:
    """
    Scaler for log-return targets.
    Scales log-returns to mean=0, std=1 so the model sees meaningful gradients.
    """

    def __init__(self, method='standard'):
        self.method = method
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler method: {method}")

    def fit(self, y):
        self.scaler.fit(y.reshape(-1, 1))
        return self

    def transform(self, y):
        return self.scaler.transform(y.reshape(-1, 1)).flatten()

    def fit_transform(self, y):
        return self.scaler.fit_transform(y.reshape(-1, 1)).flatten()

    def inverse_transform(self, y_scaled):
        return self.scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

    def get_params(self):
        if self.method == 'standard':
            return {
                'method': self.method,
                'mean':   float(self.scaler.mean_[0]),
                'std':    float(np.sqrt(self.scaler.var_[0])),
            }
        elif self.method == 'robust':
            return {
                'method': self.method,
                'center': float(self.scaler.center_[0]),
                'scale':  float(self.scaler.scale_[0]),
            }


# =============================================================================
# TARGET WINSORISATION
# =============================================================================

def winsorize_targets(y_train, y_val, percentile=99.0):
    """
    Clip extreme log-return targets to reduce the influence of flash crashes
    and pump-and-dump events on model training.

    Bounds are computed ONLY from y_train to prevent data leakage.
    y_val is clipped with the same bounds.
    """
    lower_pct = 100.0 - percentile
    upper_pct = percentile

    lower = float(np.percentile(y_train, lower_pct / 2))
    upper = float(np.percentile(y_train, 50.0 + upper_pct / 2))

    n_clipped_train = int(np.sum((y_train < lower) | (y_train > upper)))
    n_clipped_val   = int(np.sum((y_val   < lower) | (y_val   > upper)))

    y_train_w = np.clip(y_train, lower, upper)
    y_val_w   = np.clip(y_val,   lower, upper)

    pct_train = n_clipped_train / len(y_train) * 100
    pct_val   = n_clipped_val   / len(y_val)   * 100

    print(f"Winsorisation ({percentile}th pct): "
          f"bounds=[{lower:.6f}, {upper:.6f}]  "
          f"clipped train={n_clipped_train} ({pct_train:.2f}%)  "
          f"val={n_clipped_val} ({pct_val:.2f}%)")

    clip_bounds = {'lower': lower, 'upper': upper, 'percentile': percentile}
    return y_train_w, y_val_w, clip_bounds


# =============================================================================
# FLATTENING METHODS
# =============================================================================

def create_lag_features(df, features, lookback):
    """Create simple lag features (t-1, t-2, ..., t-lookback)"""
    lag_features = []
    feature_names = []
    for feature in features:
        for lag in range(1, lookback + 1):
            lag_features.append(df[feature].shift(lag))
            feature_names.append(f'{feature}_lag_{lag}')
    flat_df = pd.concat(lag_features, axis=1)
    flat_df.columns = feature_names
    return flat_df, feature_names


def create_aggregated_features(df, features, lookback):
    """
    Create aggregated temporal features (mean, std, min, max, last).
    All aggregations use window ending at t-1 to prevent data leakage.
    """
    agg_features = []
    feature_names = []
    for feature in features:
        rolled = df[feature].shift(1).rolling(window=lookback, min_periods=1)
        agg_features.append(rolled.mean());  feature_names.append(f'{feature}_mean_{lookback}')
        agg_features.append(rolled.std());   feature_names.append(f'{feature}_std_{lookback}')
        agg_features.append(rolled.min());   feature_names.append(f'{feature}_min_{lookback}')
        agg_features.append(rolled.max());   feature_names.append(f'{feature}_max_{lookback}')
        agg_features.append(df[feature].shift(1)); feature_names.append(f'{feature}_last')
    flat_df = pd.concat(agg_features, axis=1)
    flat_df.columns = feature_names
    return flat_df, feature_names


def create_hybrid_features(df, features, lookback, lag_points=None, windows=None):
    """Create hybrid features: strategic lags + multi-window aggregations"""
    if lag_points is None:
        lag_points = [1, 2, 3, 5, 10, 24]
    if windows is None:
        windows = [6, 12, 24, 48]

    lag_points = [l for l in lag_points if l <= lookback]
    windows    = [w for w in windows    if w <= lookback]

    hybrid_features = []
    feature_names   = []
    for feature in features:
        for lag in lag_points:
            hybrid_features.append(df[feature].shift(lag))
            feature_names.append(f'{feature}_lag_{lag}')
        for window in windows:
            rolled = df[feature].shift(1).rolling(window=window, min_periods=1)
            hybrid_features.append(rolled.mean()); feature_names.append(f'{feature}_mean_{window}')
            hybrid_features.append(rolled.std());  feature_names.append(f'{feature}_std_{window}')
        hybrid_features.append(df[feature].shift(1).diff(1))
        feature_names.append(f'{feature}_diff_1')
        if lookback >= 5:
            hybrid_features.append(df[feature].shift(1).diff(5))
            feature_names.append(f'{feature}_diff_5')
    flat_df = pd.concat(hybrid_features, axis=1)
    flat_df.columns = feature_names
    return flat_df, feature_names


def create_signal_features(df, features, lookback, short_windows=None,
                           long_windows=None, diff_lags=None):
    """
    Create signal-oriented features focused on momentum and trend direction.

    Instead of plain statistical summaries (mean/std/min/max), this method
    generates features that directly encode *direction* and *trend*:

      trend_short_long  -- mean(short_window) - mean(long_window)
                           positive = short-term above long-term (uptrend inside window)
      last_vs_mean      -- last value - mean(lookback)
                           how far the most recent observation sits from the window mean
      sign_fraction     -- fraction of non-zero values with positive sign in window
                           (especially informative for return_* features)
      diff_<lag>        -- first difference at strategic lags (momentum pulses)
      accel_1_5         -- diff(1) - diff(5): short-term acceleration vs medium-term

    All inputs are shifted by 1 before aggregation to prevent look-ahead leakage.

    Args:
        df:            DataFrame after prepare_features() has been called.
        features:      List of feature column names to process.
        lookback:      Window length (only windows <= lookback are used).
        short_windows: List of short window sizes for trend features.
                       Defaults to [3, 6, 12].
        long_windows:  List of long window sizes for trend features.
                       Defaults to [24, 48].  Filtered to <= lookback.
        diff_lags:     Lag steps for diff features.  Defaults to [1, 3, 5, 12].
                       Filtered to <= lookback.

    Returns:
        (flat_df, feature_names)
    """
    if short_windows is None:
        short_windows = [3, 6, 12]
    if long_windows is None:
        long_windows = [24, 48]
    if diff_lags is None:
        diff_lags = [1, 3, 5, 12]

    # Filter windows / lags that exceed lookback
    short_windows = [w for w in short_windows if w <= lookback]
    long_windows  = [w for w in long_windows  if w <= lookback]
    diff_lags     = [l for l in diff_lags     if l <= lookback]

    signal_features = []
    feature_names   = []

    for feature in features:
        col = df[feature].shift(1)   # shift to prevent leakage

        # --- trend: short mean - long mean (all pairs short x long) ---
        short_means = {}
        for sw in short_windows:
            m = col.rolling(window=sw, min_periods=1).mean()
            short_means[sw] = m

        long_means = {}
        for lw in long_windows:
            m = col.rolling(window=lw, min_periods=1).mean()
            long_means[lw] = m

        for sw, sm in short_means.items():
            for lw, lm in long_means.items():
                signal_features.append(sm - lm)
                feature_names.append(f'{feature}_trend_{sw}_{lw}')

        # --- last value vs full-window mean ---
        full_mean = col.rolling(window=lookback, min_periods=1).mean()
        signal_features.append(col - full_mean)
        feature_names.append(f'{feature}_last_vs_mean')

        # --- sign fraction: fraction of window values > 0 ---
        # Most meaningful for return_* features; harmless for others.
        sign_frac = col.rolling(window=lookback, min_periods=1).apply(
            lambda x: float(np.sum(x > 0)) / max(len(x), 1), raw=True
        )
        signal_features.append(sign_frac)
        feature_names.append(f'{feature}_sign_frac')

        # --- diffs at strategic lags ---
        diffs = {}
        for lag in diff_lags:
            d = col.diff(lag)
            diffs[lag] = d
            signal_features.append(d)
            feature_names.append(f'{feature}_diff_{lag}')

        # --- acceleration: diff(1) - diff(5) if both present ---
        if 1 in diffs and 5 in diffs:
            signal_features.append(diffs[1] - diffs[5])
            feature_names.append(f'{feature}_accel_1_5')

    flat_df = pd.concat(signal_features, axis=1)
    flat_df.columns = feature_names
    return flat_df, feature_names


def create_none_features(df, features):
    """
    No temporal flattening -- pass only the current (t-1) feature values.

    Shifts all features by 1 to prevent look-ahead leakage, identical to
    the other methods.  lookback is not used; the model sees a single
    time-step snapshot instead of a temporal window.

    Useful as a baseline to measure how much the temporal context
    (lag/agg/hybrid/signal) contributes relative to a no-history model.

    Returns flat_df with the same column names as the input features,
    and n_flat_features == n_features.
    """
    flat_df = pd.DataFrame(
        {feat: df[feat].shift(1) for feat in features}
    )
    return flat_df, list(features)


def flatten_features(df, features, lookback, method='hybrid', **kwargs):
    """Flatten sequential features into tabular format for Random Forest"""
    if method == 'lag':
        return create_lag_features(df, features, lookback)
    elif method == 'agg':
        return create_aggregated_features(df, features, lookback)
    elif method == 'hybrid':
        return create_hybrid_features(df, features, lookback,
                                      kwargs.get('lag_points'),
                                      kwargs.get('windows'))
    elif method == 'signal':
        return create_signal_features(df, features, lookback,
                                      kwargs.get('short_windows'),
                                      kwargs.get('long_windows'),
                                      kwargs.get('diff_lags'))
    elif method == 'none':
        return create_none_features(df, features)
    else:
        raise ValueError(f"Unknown flattening method: {method}")


def create_log_return_target(df, lookback, horizon):
    """
    Create log return target: log(close[t+horizon] / close[t])
    Features use window [t-lookback, t-1], current price at t, future at t+horizon.
    """
    prices   = df['close'].values
    n_samples = len(prices) - lookback - horizon
    if n_samples <= 0:
        raise ValueError(f"Insufficient data: need at least {lookback + horizon} samples")
    current = prices[lookback: lookback + n_samples]
    future  = prices[lookback + horizon: lookback + horizon + n_samples]
    return np.log(future / current).astype(np.float32)


# =============================================================================
# FEATURE SELECTION HELPER
# =============================================================================

def _resolve_features(base_features, feature_set, custom_features,
                       include_buy_pressure):
    """Resolve the feature list from prepared-data metadata."""
    if custom_features is not None:
        return [f for f in custom_features if f in base_features]

    if feature_set is not None:
        from feature_engineering import get_feature_sets
        sets = get_feature_sets(include_buy_pressure=include_buy_pressure)
        _LEGACY_ALIAS = {
            'momentum':           'xgb_momentum',
            'volatility_focused': 'xgb_volatility',
        }
        resolved_set = _LEGACY_ALIAS.get(feature_set, feature_set)
        if resolved_set not in sets:
            raise ValueError(
                f"Unknown feature set: '{feature_set}'. "
                f"Available: {sorted(sets.keys())}"
            )
        requested = sets[resolved_set] or []
        return [f for f in requested if f in base_features]

    return list(base_features)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_prepared_data(data_path, flattening_method=None, feature_set=None,
                       custom_features=None, lookback=None, horizon=None,
                       scale_target=True, target_scaler_type='standard',
                       winsorize_target=False, winsorize_percentile=99.0):
    """
    Load train/val data from a prepared directory (universal format).
    Produced by data_preparation/prepare_data.py — expects train.csv,
    scaler_features.pkl and metadata.json.
    """
    with open(os.path.join(data_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    train_csv = os.path.join(data_path, 'train.csv')
    if not os.path.exists(train_csv):
        raise ValueError(f"Missing train.csv in {data_path}")
    train_df = pd.read_csv(train_csv)

    scaler_path = os.path.join(data_path, 'scaler_features.pkl')
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(data_path, 'feature_scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Feature scaler not found in {data_path} "
            f"(tried scaler_features.pkl and feature_scaler.pkl)"
        )
    feature_scaler = joblib.load(scaler_path)

    base_features = metadata['features']
    if lookback is None:
        lookback = metadata['lookback']
    if horizon is None:
        horizon = metadata['horizon']

    has_buy_pressure = metadata.get(
        'has_buy_pressure', 'buy_pressure' in base_features
    )
    features = _resolve_features(base_features, feature_set,
                                  custom_features, has_buy_pressure)
    if not features:
        raise ValueError(
            f"No features selected (feature_set='{feature_set}', "
            f"base has {len(base_features)} features)"
        )

    if flattening_method is None:
        flattening_method = metadata.get('flattening_method', 'hybrid')

    print(f"Features: {len(features)}, Lookback: {lookback}, "
          f"Horizon: {horizon}, Flattening: {flattening_method}")

    X_flat, flat_features = flatten_features(
        train_df, features, lookback, method=flattening_method
    )
    y_all  = create_log_return_target(train_df, lookback, horizon)
    # 'none' only shifts by 1 -- drop 1 row; all other methods drop lookback rows
    drop_rows = 1 if flattening_method == 'none' else lookback
    X_flat = X_flat.iloc[drop_rows:].reset_index(drop=True).dropna()

    n      = min(len(X_flat), len(y_all))
    X_all  = X_flat.iloc[:n].values
    y_all  = y_all[:n]

    train_size = int(n * 0.8)
    X_train = X_all[:train_size];  y_train = y_all[:train_size]
    X_val   = X_all[train_size:];  y_val   = y_all[train_size:]

    clip_bounds = None
    if winsorize_target:
        y_train, y_val, clip_bounds = winsorize_targets(
            y_train, y_val, percentile=winsorize_percentile
        )

    target_scaler = None
    if scale_target:
        print(f"Scaling targets ({target_scaler_type}): "
              f"mean={y_train.mean():.6f}, std={y_train.std():.6f} -> 0, 1")
        target_scaler = TargetScaler(method=target_scaler_type)
        y_train = target_scaler.fit_transform(y_train)
        y_val   = target_scaler.transform(y_val)

    print(f"Shapes: train={X_train.shape}, val={X_val.shape}")

    data = {
        'X_train': X_train, 'y_train': y_train,
        'X_val':   X_val,   'y_val':   y_val,
        'feature_scaler': feature_scaler,
        'target_scaler':  target_scaler,
        'clip_bounds':    clip_bounds,
    }
    metadata.update({
        'flat_features':      flat_features,
        'n_flat_features':    len(flat_features),
        'n_features':         len(features),
        'features':           features,
        'lookback':           lookback,
        'horizon':            horizon,
        'flattening_method':  flattening_method,
        'has_buy_pressure':   has_buy_pressure,
        'scale_target':       scale_target,
        'target_scaler_type': target_scaler_type if scale_target else None,
        'winsorize_target':   winsorize_target,
        'winsorize_percentile': winsorize_percentile if winsorize_target else None,
        'clip_bounds':        clip_bounds,
    })
    return data, metadata


def load_test_data_only(data_path, flattening_method=None, feature_set=None,
                        custom_features=None, lookback=None, horizon=None):
    """Load only test data from a prepared directory (for rf_test.py)."""
    with open(os.path.join(data_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    test_csv = os.path.join(data_path, 'test.csv')
    if not os.path.exists(test_csv):
        raise ValueError(f"Missing test.csv in {data_path}")
    test_df = pd.read_csv(test_csv)

    scaler_path = os.path.join(data_path, 'scaler_features.pkl')
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(data_path, 'feature_scaler.pkl')
    feature_scaler = joblib.load(scaler_path)

    base_features = metadata['features']
    if lookback is None:
        lookback = metadata['lookback']
    if horizon is None:
        horizon = metadata['horizon']

    has_buy_pressure = metadata.get(
        'has_buy_pressure', 'buy_pressure' in base_features
    )
    features = _resolve_features(base_features, feature_set,
                                  custom_features, has_buy_pressure)

    if flattening_method is None:
        flattening_method = metadata.get('flattening_method', 'hybrid')

    X_test_flat, flat_features = flatten_features(
        test_df, features, lookback, method=flattening_method
    )
    y_test       = create_log_return_target(test_df, lookback, horizon)
    drop_rows    = 1 if flattening_method == 'none' else lookback
    X_test_flat  = X_test_flat.iloc[drop_rows:].reset_index(drop=True).dropna()
    min_test     = min(len(X_test_flat), len(y_test))
    X_test       = X_test_flat.iloc[:min_test].values
    y_test       = y_test[:min_test]

    print(f"Test shape: {X_test.shape}")

    data = {
        'X_test':         X_test,
        'y_test':         y_test,
        'feature_scaler': feature_scaler,
    }
    metadata.update({
        'flat_features':     flat_features,
        'n_flat_features':   len(flat_features),
        'n_features':        len(features),
        'features':          features,
        'lookback':          lookback,
        'horizon':           horizon,
        'flattening_method': flattening_method,
        'has_buy_pressure':  has_buy_pressure,
    })
    return data, metadata


def prepare_raw_data_on_the_fly(df, features=None, lookback=48, horizon=12,
                                flattening_method='hybrid',
                                feature_set=None,
                                custom_features=None,
                                feature_scaler=None,
                                scale_target=True,
                                target_scaler_type='standard',
                                winsorize_target=False,
                                winsorize_percentile=99.0):
    """
    Prepare raw CSV data on-the-fly for training (train/val only).
    Calls universal prepare_features() -> select_features() -> flatten.
    Feature scaler is fit ONLY on the training split to prevent leakage.
    """
    from feature_engineering import prepare_features, select_features

    df = prepare_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"After feature engineering: {len(df)} samples")

    if len(df) < lookback + horizon + 100:
        raise ValueError(
            f"Insufficient data after feature engineering: {len(df)} samples. "
            f"Need at least {lookback + horizon + 100}."
        )

    has_buy_pressure = 'buy_pressure' in df.columns

    if custom_features is not None:
        selected = [f for f in custom_features if f in df.columns]
    elif feature_set is not None:
        selected = select_features(
            df, feature_set=feature_set, validate=False,
            include_buy_pressure=has_buy_pressure,
        )
    elif features is not None:
        selected = [f for f in features if f in df.columns]
    else:
        selected = select_features(
            df, validate=False, include_buy_pressure=has_buy_pressure,
        )

    selected = [f for f in selected if f != 'close']
    if len(selected) < 3:
        raise ValueError(f"Too few features selected: {len(selected)}")

    print(f"Features: {len(selected)}, Flattening: {flattening_method}")

    X_flat_raw, flat_feature_names = flatten_features(
        df, selected, lookback, method=flattening_method
    )
    y_all      = create_log_return_target(df, lookback, horizon)
    drop_rows  = 1 if flattening_method == 'none' else lookback
    X_flat_raw = X_flat_raw.iloc[drop_rows:].reset_index(drop=True).dropna()

    n      = min(len(X_flat_raw), len(y_all))
    X_all  = X_flat_raw.iloc[:n]
    y_all  = y_all[:n]

    train_size   = int(n * 0.8)
    X_train_df   = X_all.iloc[:train_size]
    X_val_df     = X_all.iloc[train_size:]
    y_train      = y_all[:train_size]
    y_val        = y_all[train_size:]

    if feature_scaler is None:
        feature_scaler = StandardScaler()
        feature_scaler.fit(X_train_df)

    X_train = feature_scaler.transform(X_train_df)
    X_val   = feature_scaler.transform(X_val_df)

    clip_bounds = None
    if winsorize_target:
        y_train, y_val, clip_bounds = winsorize_targets(
            y_train, y_val, percentile=winsorize_percentile
        )

    target_scaler = None
    if scale_target:
        print(f"Scaling targets ({target_scaler_type}): "
              f"mean={y_train.mean():.6f}, std={y_train.std():.6f} -> 0, 1")
        target_scaler = TargetScaler(method=target_scaler_type)
        y_train = target_scaler.fit_transform(y_train)
        y_val   = target_scaler.transform(y_val)

    print(f"Shapes: train={X_train.shape}, val={X_val.shape}")

    data = {
        'X_train': X_train, 'y_train': y_train,
        'X_val':   X_val,   'y_val':   y_val,
        'feature_scaler': feature_scaler,
        'target_scaler':  target_scaler,
        'clip_bounds':    clip_bounds,
    }
    metadata = {
        'model_type':         'random_forest',
        'target_type':        'log_return',
        'features':           selected,
        'flat_features':      flat_feature_names,
        'n_features':         len(selected),
        'n_flat_features':    len(flat_feature_names),
        'lookback':           lookback,
        'horizon':            horizon,
        'flattening_method':  flattening_method,
        'has_buy_pressure':   has_buy_pressure,
        'scale_target':       scale_target,
        'target_scaler_type': target_scaler_type if scale_target else None,
        'winsorize_target':   winsorize_target,
        'winsorize_percentile': winsorize_percentile if winsorize_target else None,
        'clip_bounds':        clip_bounds,
    }
    return data, metadata


def load_data(data_path, scale_target=True, target_scaler_type='standard',
              winsorize_target=False, winsorize_percentile=99.0, **kwargs):
    """
    Unified entry point -- load data from prepared directory or raw CSV.
    kwargs forwarded: flattening_method, feature_set, custom_features,
                      lookback, horizon
    """
    if os.path.isdir(data_path):
        if not os.path.exists(os.path.join(data_path, 'metadata.json')):
            raise ValueError(f"Directory without metadata.json: {data_path}")
        data, metadata = load_prepared_data(
            data_path,
            flattening_method=kwargs.get('flattening_method'),
            feature_set=kwargs.get('feature_set'),
            custom_features=kwargs.get('custom_features'),
            lookback=kwargs.get('lookback'),
            horizon=kwargs.get('horizon'),
            scale_target=scale_target,
            target_scaler_type=target_scaler_type,
            winsorize_target=winsorize_target,
            winsorize_percentile=winsorize_percentile,
        )
        return data, True, metadata

    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records from CSV")
        data, metadata = prepare_raw_data_on_the_fly(
            df,
            feature_set=kwargs.get('feature_set'),
            custom_features=kwargs.get('custom_features'),
            lookback=kwargs.get('lookback', 48),
            horizon=kwargs.get('horizon', 12),
            flattening_method=kwargs.get('flattening_method', 'hybrid'),
            scale_target=scale_target,
            target_scaler_type=target_scaler_type,
            winsorize_target=winsorize_target,
            winsorize_percentile=winsorize_percentile,
        )
        return data, False, metadata

    else:
        raise ValueError(
            f"Invalid data path: {data_path}. "
            f"Expected a prepared directory or a .csv file."
        )
