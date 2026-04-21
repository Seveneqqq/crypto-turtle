"""
Flattening utilities for XGBoost model.

Transforms sequential (time-series) features into a flat tabular format
suitable for XGBoost. Each method produces a DataFrame of engineered columns
from a lookback window, shifted by 1 to prevent look-ahead leakage.

Available methods:
    lag     — simple lag features (t-1, t-2, ..., t-lookback)
    agg     — aggregated statistics (mean, std, min, max, last)
    hybrid  — strategic lags + multi-window aggregations + diffs
    signal  — momentum/trend-oriented (trend crossovers, sign fraction, accel)
    none    — single snapshot at t-1 (baseline, no temporal context)
"""
import numpy as np
import pandas as pd


# =============================================================================
# LAG FLATTENING
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


# =============================================================================
# AGGREGATED FLATTENING
# =============================================================================

def create_aggregated_features(df, features, lookback):
    """
    Create aggregated temporal features (mean, std, min, max, last).
    All aggregations use window ending at t-1 to prevent data leakage.
    """
    agg_features = []
    feature_names = []

    for feature in features:
        rolled = df[feature].shift(1).rolling(window=lookback, min_periods=1)

        agg_features.append(rolled.mean())
        feature_names.append(f'{feature}_mean_{lookback}')

        agg_features.append(rolled.std())
        feature_names.append(f'{feature}_std_{lookback}')

        agg_features.append(rolled.min())
        feature_names.append(f'{feature}_min_{lookback}')

        agg_features.append(rolled.max())
        feature_names.append(f'{feature}_max_{lookback}')

        agg_features.append(df[feature].shift(1))
        feature_names.append(f'{feature}_last')

    flat_df = pd.concat(agg_features, axis=1)
    flat_df.columns = feature_names
    return flat_df, feature_names


# =============================================================================
# HYBRID FLATTENING
# =============================================================================

def create_hybrid_features(df, features, lookback, lag_points=None, windows=None):
    """Create hybrid features: strategic lags + multi-window aggregations"""
    if lag_points is None:
        lag_points = [1, 2, 3, 5, 10, 24]
    if windows is None:
        windows = [6, 12, 24, 48]

    lag_points = [l for l in lag_points if l <= lookback]
    windows = [w for w in windows if w <= lookback]

    hybrid_features = []
    feature_names = []

    for feature in features:
        for lag in lag_points:
            hybrid_features.append(df[feature].shift(lag))
            feature_names.append(f'{feature}_lag_{lag}')

        for window in windows:
            rolled = df[feature].shift(1).rolling(window=window, min_periods=1)
            hybrid_features.append(rolled.mean())
            feature_names.append(f'{feature}_mean_{window}')
            hybrid_features.append(rolled.std())
            feature_names.append(f'{feature}_std_{window}')

        hybrid_features.append(df[feature].shift(1).diff(1))
        feature_names.append(f'{feature}_diff_1')

        if lookback >= 5:
            hybrid_features.append(df[feature].shift(1).diff(5))
            feature_names.append(f'{feature}_diff_5')

    flat_df = pd.concat(hybrid_features, axis=1)
    flat_df.columns = feature_names
    return flat_df, feature_names


# =============================================================================
# NONE FLATTENING (baseline)
# =============================================================================

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


# =============================================================================
# SIGNAL FLATTENING
# =============================================================================

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


# =============================================================================
# DISPATCHER
# =============================================================================

def flatten_features(df, features, lookback, method='hybrid', **kwargs):
    """Flatten sequential features into tabular format for XGBoost"""
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
