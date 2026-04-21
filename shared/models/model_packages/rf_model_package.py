"""
RFModelPackage -- standalone inference package for Random Forest crypto models.

Produces probabilistic BUY / SELL signals from a trained Random Forest
log-return model.  Signal model is identical to XGBModelPackage (Approach A):

    mu    = model prediction (log-return, after inverse_transform)
    sigma = rolling std of log-returns from close prices (market volatility)
    T     = threshold (minimum economically meaningful move)

    signal_buy  = P(r >  T) = (1 - Phi((T  - mu) / sigma)) * 100   [0-100]
    signal_sell = P(r < -T) = Phi((-T - mu) / sigma) * 100          [0-100]

    BUY + SELL < 100 always (gap = P(-T <= r <= T) = "hold zone").

Inputs accepted:
    'raw'      -- raw OHLCV from Binance (open_time, open, high, low, close, volume)
    'prepared' -- prepared CSV from data_preparation (features already scaled,
                  close is raw price)
    'auto'     -- auto-detect based on column presence (default)

Key differences vs XGBModelPackage:
    - loads rf_model.pkl  (joblib) instead of xgb_model.json (XGBoost Booster)
    - model.predict(X)    -- no DMatrix, no iteration_range
    - no iter_range / saved_iteration logic
    - feature importance via model.feature_importances_ (MDI), not gain/weight

Usage:
    from rf_model_package import RFModelPackage, SignalConfig

    cfg = SignalConfig(threshold=0.001, window_sigma=100)
    pkg = RFModelPackage('rf_models/R001_btc_rf_model_B001_1503', signal_config=cfg)

    # Batch (simulation / backtest)
    df_signals = pkg.predict_batch(df)
    # columns: timestamp, mu, sigma, signal_buy, signal_sell

    # Live (one signal for latest candle)
    signal = pkg.predict_live(df_buffer)
    # dict: {timestamp, mu, sigma, signal_buy, signal_sell}

    # Model info
    info = pkg.get_model_info()

    # Minimum rows needed for predict_live
    n = pkg.min_rows_required
"""

import os
import sys
import json
import dataclasses
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, RobustScaler

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Feature engineering (resolves root feature_engineering.py via local copy)
# ---------------------------------------------------------------------------
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from feature_engineering import prepare_features  # noqa: E402


# ===========================================================================
# CONFIG DATACLASS
# ===========================================================================

@dataclasses.dataclass
class SignalConfig:
    """
    Parameters for the probabilistic signal generator.

    threshold:    Minimum log-return to trigger a signal.
                  Should cover trading fees + spread.
                  Typical range for BTC/ETH 1m: 0.0005 - 0.005
                  Default 0.001 ~= 0.1% (covers ~0.1% fee round-trip).

    window_sigma: Rolling window (in candles) for estimating market volatility.
                  Larger = more stable but slower to react.
                  Recommended for 1m data: 50 - 200.
                  Default 100 = ~100 minutes of history.
    """
    threshold:    float = 0.001
    window_sigma: int   = 100


# ===========================================================================
# FLATTENING  (self-contained -- no dependency on rf_model/common/data_utils)
# ===========================================================================

def _create_aggregated_features(df: pd.DataFrame, features: list, lookback: int):
    """Aggregate temporal features (mean, std, min, max, last) over lookback window."""
    agg_list = []
    names    = []
    for feat in features:
        rolled = df[feat].shift(1).rolling(window=lookback, min_periods=1)
        agg_list += [
            rolled.mean(), rolled.std(), rolled.min(), rolled.max(),
            df[feat].shift(1),
        ]
        names += [
            f'{feat}_mean_{lookback}', f'{feat}_std_{lookback}',
            f'{feat}_min_{lookback}',  f'{feat}_max_{lookback}',
            f'{feat}_last',
        ]
    flat = pd.concat(agg_list, axis=1)
    flat.columns = names
    return flat, names


def _create_lag_features(df: pd.DataFrame, features: list, lookback: int):
    """Create simple lag features t-1 ... t-lookback."""
    lag_list = []
    names    = []
    for feat in features:
        for lag in range(1, lookback + 1):
            lag_list.append(df[feat].shift(lag))
            names.append(f'{feat}_lag_{lag}')
    flat = pd.concat(lag_list, axis=1)
    flat.columns = names
    return flat, names


def _create_hybrid_features(df: pd.DataFrame, features: list, lookback: int,
                             lag_points=None, windows=None):
    """Strategic lags + multi-window rolling stats + diffs."""
    if lag_points is None:
        lag_points = [1, 2, 3, 5, 10, 24]
    if windows is None:
        windows = [6, 12, 24, 48]
    lag_points = [l for l in lag_points if l <= lookback]
    windows    = [w for w in windows    if w <= lookback]

    parts = []
    names = []
    for feat in features:
        for lag in lag_points:
            parts.append(df[feat].shift(lag))
            names.append(f'{feat}_lag_{lag}')
        for w in windows:
            rolled = df[feat].shift(1).rolling(window=w, min_periods=1)
            parts += [rolled.mean(), rolled.std()]
            names += [f'{feat}_mean_{w}', f'{feat}_std_{w}']
        parts.append(df[feat].shift(1).diff(1))
        names.append(f'{feat}_diff_1')
        if lookback >= 5:
            parts.append(df[feat].shift(1).diff(5))
            names.append(f'{feat}_diff_5')

    flat = pd.concat(parts, axis=1)
    flat.columns = names
    return flat, names


def _create_signal_features(df: pd.DataFrame, features: list, lookback: int,
                            short_windows=None, long_windows=None,
                            diff_lags=None):
    """
    Create signal-oriented features focused on momentum and trend direction.

    Generated features per input feature:
      trend_S_L       -- mean(S) - mean(L) for each (short, long) window pair
                         positive = short-term above long-term (uptrend inside window)
      last_vs_mean    -- last value - mean(lookback)
                         how far the most recent observation sits from window mean
      sign_frac       -- fraction of window values > 0
                         most informative for return_* features
      diff_<lag>      -- first difference at strategic lags (momentum pulses)
      accel_1_5       -- diff(1) - diff(5): short-term acceleration vs medium-term
                         (only generated when both lag 1 and lag 5 are present)

    All inputs are shifted by 1 before aggregation to prevent look-ahead leakage.
    Windows and lags exceeding lookback are silently skipped.
    """
    if short_windows is None:
        short_windows = [3, 6, 12]
    if long_windows is None:
        long_windows = [24, 48]
    if diff_lags is None:
        diff_lags = [1, 3, 5, 12]

    short_windows = [w for w in short_windows if w <= lookback]
    long_windows  = [w for w in long_windows  if w <= lookback]
    diff_lags     = [l for l in diff_lags     if l <= lookback]

    parts = []
    names = []

    for feat in features:
        col = df[feat].shift(1)   # shift-1: prevent look-ahead leakage

        # --- trend: mean(short) - mean(long) for every (short, long) pair ---
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
                parts.append(sm - lm)
                names.append(f'{feat}_trend_{sw}_{lw}')

        # --- last value vs full-window mean ---
        full_mean = col.rolling(window=lookback, min_periods=1).mean()
        parts.append(col - full_mean)
        names.append(f'{feat}_last_vs_mean')

        # --- sign fraction: fraction of window values > 0 ---
        sign_frac = col.rolling(window=lookback, min_periods=1).apply(
            lambda x: float(np.sum(x > 0)) / max(len(x), 1), raw=True
        )
        parts.append(sign_frac)
        names.append(f'{feat}_sign_frac')

        # --- diffs at strategic lags ---
        diffs = {}
        for lag in diff_lags:
            d = col.diff(lag)
            diffs[lag] = d
            parts.append(d)
            names.append(f'{feat}_diff_{lag}')

        # --- acceleration: diff(1) - diff(5) when both lags are present ---
        if 1 in diffs and 5 in diffs:
            parts.append(diffs[1] - diffs[5])
            names.append(f'{feat}_accel_1_5')

    flat = pd.concat(parts, axis=1)
    flat.columns = names
    return flat, names


def _create_none_features(df: pd.DataFrame, features: list):
    """No temporal flattening -- pass only the current (t-1) feature values."""
    flat_df = pd.DataFrame(
        {feat: df[feat].shift(1) for feat in features}
    )
    return flat_df, list(features)


def _flatten_features(df: pd.DataFrame, features: list, lookback: int,
                      method: str):
    """Dispatch to the correct flattening method."""
    if method == 'agg':
        return _create_aggregated_features(df, features, lookback)
    elif method == 'lag':
        return _create_lag_features(df, features, lookback)
    elif method == 'hybrid':
        return _create_hybrid_features(df, features, lookback)
    elif method == 'signal':
        return _create_signal_features(df, features, lookback)
    elif method == 'none':
        return _create_none_features(df, features)
    else:
        raise ValueError(
            f"Unknown flattening method: '{method}'. "
            f"Expected 'agg', 'lag', 'hybrid', 'signal', or 'none'."
        )


# ===========================================================================
# TARGET SCALER  (standalone -- no rf_model/common dependency)
# ===========================================================================

class _TargetScaler:
    """
    Thin wrapper around sklearn scalers for 1-D log-return arrays.
    Reconstructed from target_scaler_params stored in model_config.json --
    no pickle module dependency on rf_model.common.data_utils.TargetScaler.
    """

    def __init__(self, method: str = 'robust'):
        self.method = method
        if method == 'robust':
            self._scaler = RobustScaler()
        elif method == 'standard':
            self._scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler method: '{method}'")

    @classmethod
    def from_params(cls, params: dict) -> '_TargetScaler':
        """Reconstruct scaler from target_scaler_params in model_config.json."""
        obj = cls(method=params['method'])
        if params['method'] == 'robust':
            obj._scaler.center_        = np.array([params['center']])
            obj._scaler.scale_         = np.array([params['scale']])
            obj._scaler.n_features_in_ = 1
        elif params['method'] == 'standard':
            obj._scaler.mean_            = np.array([params['mean']])
            obj._scaler.scale_           = np.array([params['std']])
            obj._scaler.var_             = np.array([params['std'] ** 2])
            obj._scaler.n_features_in_   = 1
        return obj

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return self._scaler.inverse_transform(y.reshape(-1, 1)).flatten()


# ===========================================================================
# VALIDATION HELPERS
# ===========================================================================

_RAW_REQUIRED  = {'open_time', 'open', 'high', 'low', 'close', 'volume'}
_PREP_REQUIRED = {'open_time', 'close'}
# Minimum warmup rows for prepare_features (return_60 needs 60, EWM ~3x26=78)
_FE_WARMUP = 100


def _detect_mode(df: pd.DataFrame, config: dict) -> str:
    """Return 'prepared' if feature columns already present, else 'raw'."""
    first_feat = config['features'][0]
    return 'prepared' if first_feat in df.columns else 'raw'


def _validate(df: pd.DataFrame, mode: str, config: dict,
              window_sigma: int) -> None:
    """
    Raise ValueError with a descriptive message if df is not suitable.
    Checks are intentionally fast (no heavy iteration).
    For 'none' flattening method, lookback is not used (only shift(1)),
    so the effective minimum lookback contribution is 1.
    """
    method   = config.get('flattening_method', 'hybrid')
    lookback = config['lookback']
    # For 'none', only 1 row of lookback context is needed (shift by 1)
    effective_lb = 1 if method == 'none' else lookback

    if mode == 'raw':
        missing = _RAW_REQUIRED - set(df.columns)
        if missing:
            raise ValueError(
                f"[raw mode] Missing required columns: {sorted(missing)}\n"
                f"Expected: {sorted(_RAW_REQUIRED)}"
            )
        min_rows = effective_lb + window_sigma + _FE_WARMUP
        if len(df) < min_rows:
            raise ValueError(
                f"[raw mode] Insufficient rows: got {len(df)}, "
                f"need >= {min_rows} "
                f"(effective_lookback={effective_lb} + "
                f"window_sigma={window_sigma} + warmup={_FE_WARMUP})"
            )

    elif mode == 'prepared':
        missing_base = _PREP_REQUIRED - set(df.columns)
        if missing_base:
            raise ValueError(
                f"[prepared mode] Missing base columns: {sorted(missing_base)}"
            )
        missing_feats = [f for f in config['features'] if f not in df.columns]
        if missing_feats:
            preview = missing_feats[:5]
            extra   = (f' ... (+{len(missing_feats)-5} more)'
                       if len(missing_feats) > 5 else '')
            raise ValueError(
                f"[prepared mode] Missing feature columns: {preview}{extra}\n"
                f"Model expects {len(config['features'])} features from config."
            )
        min_rows = effective_lb + window_sigma
        if len(df) < min_rows:
            raise ValueError(
                f"[prepared mode] Insufficient rows: got {len(df)}, "
                f"need >= {min_rows} "
                f"(effective_lookback={effective_lb} + "
                f"window_sigma={window_sigma})"
            )
    else:
        raise ValueError(
            f"Unknown mode: '{mode}'. Use 'raw', 'prepared', or 'auto'."
        )

    # Shared checks
    if df['close'].isna().any():
        raise ValueError("Column 'close' contains NaN values.")
    if (df['close'] <= 0).any():
        raise ValueError("Column 'close' contains zero or negative values.")


# ===========================================================================
# SIGNAL CALCULATION  (vectorised)
# ===========================================================================

def _compute_sigma(close: pd.Series, window: int) -> pd.Series:
    """Rolling std of log-returns. First (window-1) values are NaN."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=window, min_periods=window).std()


def _compute_signals(mu: np.ndarray, sigma: np.ndarray,
                     threshold: float) -> tuple:
    """
    Vectorised signal calculation.

    signal_buy  = P(r >  T | mu, sigma) * 100
    signal_sell = P(r < -T | mu, sigma) * 100

    Values outside [0, 100] are clipped (numerical safety).
    """
    sigma_safe  = np.maximum(sigma, 1e-8)
    signal_buy  = (1.0 - norm.cdf((threshold  - mu) / sigma_safe)) * 100.0
    signal_sell = norm.cdf((-threshold - mu) / sigma_safe) * 100.0
    signal_buy  = np.clip(signal_buy,  0.0, 100.0)
    signal_sell = np.clip(signal_sell, 0.0, 100.0)
    return signal_buy, signal_sell


# ===========================================================================
# MAIN CLASS
# ===========================================================================

class RFModelPackage:
    """
    Standalone inference + signal package for a trained Random Forest
    log-return model.

    Parameters
    ----------
    model_dir : str
        Path to model directory produced by rf_train.py.
        Must contain: rf_model.pkl, feature_scaler.pkl, model_config.json
        Optional:     target_scaler.pkl (reconstructed from model_config if
                      target_scaler_params present in JSON)

    signal_config : SignalConfig, optional
        Signal generation parameters. Uses defaults if None.
    """

    def __init__(self, model_dir: str,
                 signal_config: Optional[SignalConfig] = None):
        self._model_dir = model_dir
        self._sig_cfg   = signal_config or SignalConfig()

        self._config              = self._load_config()
        self._model               = self._load_model()
        self._feature_scaler      = self._load_feature_scaler()
        self._target_scaler       = self._load_target_scaler()
        self._scaler_feature_cols = self._resolve_scaler_feature_cols()

    # ------------------------------------------------------------------
    # LOADING
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        path = os.path.join(self._model_dir, 'model_config.json')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"model_config.json not found in: {self._model_dir}"
            )
        with open(path, 'r') as f:
            return json.load(f)

    def _load_model(self):
        """Load Random Forest model from rf_model.pkl (joblib)."""
        path = os.path.join(self._model_dir, 'rf_model.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"rf_model.pkl not found in: {self._model_dir}"
            )
        return joblib.load(path)

    def _load_feature_scaler(self):
        """Load feature scaler. Tries both naming conventions."""
        for name in ('feature_scaler.pkl', 'scaler_features.pkl'):
            path = os.path.join(self._model_dir, name)
            if os.path.exists(path):
                return joblib.load(path)
        raise FileNotFoundError(
            f"Feature scaler not found in {self._model_dir} "
            f"(tried feature_scaler.pkl and scaler_features.pkl)"
        )

    def _load_target_scaler(self) -> Optional['_TargetScaler']:
        """
        Load target scaler.

        Priority:
          1. target_scaler_params from model_config.json -- always preferred;
             requires no external module, immune to pickle compatibility issues.
          2. target_scaler.pkl -- attempted only when params are absent.
             rf_model.common.data_utils.TargetScaler must be importable;
             if not, a clear RuntimeError is raised.
        """
        if not self._config.get('scale_target', False):
            return None

        # Priority 1: reconstruct from JSON params -- no pickle, no module deps
        params = self._config.get('target_scaler_params')
        if params:
            return _TargetScaler.from_params(params)

        # Priority 2: load .pkl as fallback (models without params in config)
        pkl_path = os.path.join(self._model_dir, 'target_scaler.pkl')
        if os.path.exists(pkl_path):
            try:
                raw = joblib.load(pkl_path)
            except (ModuleNotFoundError, ImportError) as exc:
                raise RuntimeError(
                    f"Cannot load target_scaler.pkl ({exc}) and "
                    f"'target_scaler_params' is missing from model_config.json.\n"
                    f"Re-save the model with rf_train.py to embed scaler params."
                ) from exc
            # Wrap a bare sklearn scaler if needed
            if isinstance(raw, _TargetScaler):
                return raw
            wrapper         = _TargetScaler.__new__(_TargetScaler)
            wrapper.method  = self._config.get('target_scaler_type', 'robust')
            wrapper._scaler = raw
            return wrapper

        return None

    def _resolve_scaler_feature_cols(self) -> Optional[list]:
        """
        Determine the ordered list of feature columns that feature_scaler was
        fitted on (needed in 'raw' mode to pass the full matrix to
        scaler.transform() in the correct column order).

        Resolution order:
          1. scaler_feature_cols from model_config.json  (self-contained)
          2. Prepared data metadata.json referenced by train_path
          3. model_config['features']  (if count matches scaler)
          4. None  (raw mode will warn + best-effort scale)
        """
        n = self._feature_scaler.n_features_in_

        # 1. Directly from model_config.json
        cols = self._config.get('scaler_feature_cols')
        if cols and len(cols) == n:
            return cols

        # 2. Try prepared data metadata.json
        train_path = self._config.get('train_path', '')
        if train_path:
            candidates = [
                train_path,
                os.path.join(self._model_dir, '..', '..', train_path),
                os.path.join(self._model_dir, '..', train_path),
                os.path.join(os.path.dirname(self._model_dir), train_path),
            ]
            for base in candidates:
                meta_path = os.path.normpath(
                    os.path.join(base, 'metadata.json')
                )
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                        cols = meta.get('features', [])
                        if len(cols) == n:
                            return cols
                    except Exception:
                        pass

        # 3. Fallback: model config features
        model_feats = self._config.get('features', [])
        if len(model_feats) == n:
            return model_feats

        return None

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    @property
    def min_rows_required(self) -> int:
        """
        Minimum number of rows needed in the input DataFrame for predict_live.
        For 'prepared' mode: lookback + window_sigma.
        For 'raw' mode:      lookback + window_sigma + FE warmup (100).
        For 'none' method:   1 + window_sigma (+FE warmup for raw mode).
        Returns the larger value (safe for both prepared and raw modes).
        """
        method = self._config.get('flattening_method', 'hybrid')
        lb = 1 if method == 'none' else self._config['lookback']
        ws = self._sig_cfg.window_sigma
        return lb + ws + _FE_WARMUP

    def get_model_info(self) -> dict:
        """Return key model metadata."""
        c = self._config
        return {
            'model_id':            c.get('model_id',          'unknown'),
            'data_id':             c.get('data_id',           'unknown'),
            'symbol':              c.get('symbol',            'unknown'),
            'lookback':            c.get('lookback'),
            'horizon':             c.get('horizon'),
            'n_features':          c.get('n_features'),
            'n_flat_features':     c.get('n_flat_features'),
            'flattening_method':   c.get('flattening_method'),
            'feature_set':         c.get('feature_set'),
            'criterion':           c.get('params', {}).get('criterion', 'N/A'),
            'n_estimators':        c.get('params', {}).get('n_estimators'),
            'scale_target':        c.get('scale_target'),
            'target_scaler_type':  c.get('target_scaler_type'),
            'oob_score':           c.get('oob_score'),
            'train_metrics':       c.get('train_metrics'),
            'val_metrics':         c.get('val_metrics'),
            'timestamp':           c.get('timestamp'),
            'dir_acc_threshold':   c.get('dir_acc_threshold'),
            'signal_threshold':    self._sig_cfg.threshold,
            'signal_window_sigma': self._sig_cfg.window_sigma,
        }

    def predict_batch(self, df: pd.DataFrame,
                      data_mode: str = 'auto') -> pd.DataFrame:
        """
        Process an entire DataFrame and return signals for every valid sample.

        Parameters
        ----------
        df : pd.DataFrame
            Input data. See module docstring for column requirements.
        data_mode : str
            'raw' | 'prepared' | 'auto' (default).

        Returns
        -------
        pd.DataFrame with columns:
            timestamp    -- open_time of the current candle (t)
            mu           -- predicted log-return (after inverse_transform)
            sigma        -- rolling volatility at candle t
            signal_buy   -- P(r > threshold) * 100  [0-100]
            signal_sell  -- P(r < -threshold) * 100 [0-100]

        Rows where sigma is NaN (first window_sigma-1 candles) are dropped.
        """
        mode = (_detect_mode(df, self._config)
                if data_mode == 'auto' else data_mode)
        _validate(df, mode, self._config, self._sig_cfg.window_sigma)

        df_work = self._prepare_df(df, mode)
        mu      = self._predict_mu(df_work)
        sigma_aligned, timestamps = self._compute_aligned_sigma(
            df_work, len(mu)
        )
        return self._build_output(mu, sigma_aligned, timestamps)

    def predict_live(self, df: pd.DataFrame,
                     data_mode: str = 'auto') -> dict:
        """
        Produce a single signal for the latest candle (live / streaming mode).

        Internally calls predict_batch and returns the last row as a dict.
        The caller must maintain a rolling buffer of at least min_rows_required
        rows.

        Parameters
        ----------
        df : pd.DataFrame
            Rolling buffer (latest candles), same format as predict_batch.
        data_mode : str
            'raw' | 'prepared' | 'auto' (default).

        Returns
        -------
        dict: {timestamp, mu, sigma, signal_buy, signal_sell}
        """
        result_df = self.predict_batch(df, data_mode=data_mode)
        if result_df.empty:
            raise RuntimeError(
                "predict_live: no valid signals produced. "
                f"Provide at least {self.min_rows_required} rows."
            )
        row = result_df.iloc[-1]
        return {
            'timestamp':   row['timestamp'],
            'mu':          float(row['mu']),
            'sigma':       float(row['sigma']),
            'signal_buy':  float(row['signal_buy']),
            'signal_sell': float(row['signal_sell']),
        }

    # ------------------------------------------------------------------
    # INTERNAL PIPELINE
    # ------------------------------------------------------------------

    def _prepare_df(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        Return a clean DataFrame with scaled feature columns and raw close prices.
        The returned df has a clean integer index (reset_index).
        """
        features = self._config['features']

        if mode == 'raw':
            df_feat = prepare_features(df.copy())
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan)

            # Drop rows where any model feature is NaN (buy_pressure may be
            # absent if taker_buy column not in input -- that is acceptable).
            available_model_feats = [
                f for f in features if f in df_feat.columns
            ]
            df_feat = df_feat.dropna(
                subset=available_model_feats
            ).reset_index(drop=True)
            df_feat = df_feat.copy()

            # Scale features.
            # feature_scaler was fit on the full base feature set at training
            # time; we must pass columns in the exact order the scaler expects.
            scaler_cols = self._scaler_feature_cols
            if scaler_cols is not None:
                # Fill any column the scaler needs that raw OHLCV cannot produce
                for col in scaler_cols:
                    if col not in df_feat.columns:
                        df_feat[col] = 0.0
                df_feat[scaler_cols] = self._feature_scaler.transform(
                    df_feat[scaler_cols]
                )
            else:
                warnings.warn(
                    "Could not resolve scaler column order from prepared data "
                    "metadata. Scaling only available model features -- "
                    "predictions may be inaccurate.",
                    RuntimeWarning,
                )
                if available_model_feats:
                    df_feat[available_model_feats] = (
                        self._feature_scaler.transform(
                            df_feat[available_model_feats]
                        )
                    )

        else:   # prepared
            # Features already scaled; close is raw price
            df_feat = df.copy().reset_index(drop=True)

        return df_feat

    def _predict_mu(self, df_work: pd.DataFrame) -> np.ndarray:
        """
        Run flattening -> RF predict -> inverse_transform -> mu array.

        RF predict() accepts plain numpy arrays directly -- no DMatrix,
        no iteration_range.
        """
        features = self._config['features']
        lookback = self._config['lookback']
        method   = self._config['flattening_method']

        X_flat, _ = _flatten_features(df_work, features, lookback, method)

        # For 'none', only shift(1) is applied -- drop just 1 leading row.
        # For all other methods the first `lookback` rows have no complete
        # window and must be dropped.
        drop_rows = 1 if method == 'none' else lookback
        X_flat = X_flat.iloc[drop_rows:].reset_index(drop=True).dropna()
        X_np   = X_flat.values.astype(np.float32)

        y_scaled = self._model.predict(X_np)   # plain ndarray, no DMatrix

        if self._target_scaler is not None:
            mu = self._target_scaler.inverse_transform(y_scaled)
        else:
            mu = y_scaled

        return mu.astype(np.float64)

    def _compute_aligned_sigma(self, df_work: pd.DataFrame,
                                n_mu: int) -> tuple:
        """
        Compute sigma aligned with mu predictions.

        For prediction sample i (0-indexed), the RF window covers df_work rows
        [i .. i+lookback-1], so the "current candle" index in df_work is
        i + lookback - 1.  Therefore:
            sigma_aligned[i] = sigma_series.iloc[i + lookback - 1]

        Returns (sigma_aligned, timestamps), both length == n_mu.
        NaN sigma values are preserved; caller drops them via _build_output.
        """
        window   = self._sig_cfg.window_sigma
        lookback = self._config['lookback']

        sigma_series = _compute_sigma(df_work['close'], window)
        timestamps   = (df_work['open_time']
                        if 'open_time' in df_work.columns
                        else pd.RangeIndex(len(df_work)))

        # Map sample index -> df_work row index.
        # For 'none' method only shift(1) is used (drop_rows=1), so the
        # current candle for sample i is at df_work index i + 1 - 1 = i.
        # For all other methods it is i + lookback - 1.
        method = self._config['flattening_method']
        offset = 0 if method == 'none' else (lookback - 1)
        current_candle_indices = np.arange(n_mu) + offset
        max_idx                = len(df_work) - 1
        current_candle_indices = np.clip(current_candle_indices, 0, max_idx)

        sigma_aligned = sigma_series.iloc[current_candle_indices].values
        ts_aligned    = timestamps.iloc[current_candle_indices].values

        return sigma_aligned, ts_aligned

    def _build_output(self, mu: np.ndarray, sigma: np.ndarray,
                      timestamps: np.ndarray) -> pd.DataFrame:
        """Compute signals and return clean output DataFrame (NaN sigma dropped)."""
        signal_buy, signal_sell = _compute_signals(
            mu, sigma, self._sig_cfg.threshold
        )
        out = pd.DataFrame({
            'timestamp':   timestamps,
            'mu':          mu,
            'sigma':       sigma,
            'signal_buy':  signal_buy,
            'signal_sell': signal_sell,
        })
        out = out.dropna(subset=['sigma']).reset_index(drop=True)
        return out
