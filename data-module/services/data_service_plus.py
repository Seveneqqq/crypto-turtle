import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

from .data_service import DataService


PREPROCESSING_METHODS = {"RAW", "Z-SCORE", "ROBUST", "MINMAX", "ROLLING_Z"}

EXCLUDE_COLUMNS_REALTIME = [
    "open_time", "open", "high", "low", "close", "close_time",
]

# Pozwala compute_features_for_realtime obliczać tylko to co potrzebne
_FEATURE_DEPS: dict[str, set[str]] = {
    # ret1 i lagi
    "ret1":        set(),
    "ret1_lag1":   {"ret1"},
    "ret1_lag2":   {"ret1"},
    "ret1_lag5":   {"ret1"},
    "ret1_lag10":  {"ret1"},
    # rolling
    "roll_mean_10": set(),
    "roll_mean_50": set(),
    "roll_std_10":  set(),
    "roll_std_50":  set(),
    # macd
    "macd":     set(),
    "macd_sig": {"macd"},
    # set1
    "ma_5":        set(),
    "ma_10":       set(),
    "ma_20":       set(),
    "sma_10":      set(),
    "sma_30":      set(),
    "cumvol":      set(),
    "volume_norm": set(),
    "std_20":      set(),
    # set2
    "RSI_14": set(),
    "EMA_20":  set(),
    "EMA_50":  set(),
    # set3
    "BB_upper":    set(),
    "BB_lower":    set(),
    "Momentum_10": set(),
    "ROC_10":      set(),
    # set4
    "ATR_14": set(),
    "VWAP":   set(),
    "ROC_5":  set(),
    # target
    "target": set(),
}


def _resolve_required(features: list[str]) -> set[str]:
    """Zwraca pełną listę features do obliczenia (z dependencjami)."""
    needed = set()
    for f in features:
        needed.add(f)
        needed.update(_FEATURE_DEPS.get(f, set()))
    return needed


# ============================================================
# ROLLING Z-SCORE
# ============================================================

class RollingZScore(BaseEstimator, TransformerMixin):
    def __init__(self, min_periods=50):
        self.min_periods = min_periods
        self._train_X = None

    def fit(self, X, y=None):
        self._train_X = pd.DataFrame(X).astype(float).values
        return self

    def transform(self, X):
        X_new = pd.DataFrame(X).astype(float)
        if self._train_X is not None:
            concat = pd.DataFrame(np.vstack([self._train_X, X_new.values])).astype(float)
            mu = concat.expanding(min_periods=self.min_periods).mean()
            sd = concat.expanding(min_periods=self.min_periods).std().replace(0, np.nan)
            return (concat - mu).divide(sd).fillna(0.0).values[len(self._train_X):]
        else:
            mu = X_new.expanding(min_periods=self.min_periods).mean()
            sd = X_new.expanding(min_periods=self.min_periods).std().replace(0, np.nan)
            return (X_new - mu).divide(sd).fillna(0.0).values


def transform_rolling_no_leak(Xtr_raw, Xte_raw):
    rz = RollingZScore()
    Xtr = rz.fit_transform(Xtr_raw)
    Xall = RollingZScore().fit_transform(np.vstack([Xtr_raw, Xte_raw]))
    return Xtr, Xall[len(Xtr_raw):]


# ============================================================
# DATA SERVICE BETTER
# ============================================================

class DataServicePlus(DataService):
    """
    Rozszerzenie DataService o metody potrzebne w realtime:
    - fit_scaler_from_file           — fit na pliku treningowym
    - compute_features_for_realtime  — statyczna, oblicza tylko wymagane cechy
    - apply_preprocessing_realtime   — statyczna, transform na oknie 100 świec
    """

    def __init__(self):
        super().__init__()
        proc_base_dir = os.path.join(self.base_dir, "shared", "files")
        self.proc_dirs = {
            "trainmodel": os.path.join(proc_base_dir, "processed_trainmodel_data"),
            "testmodel":  os.path.join(proc_base_dir, "processed_testmodel_data"),
            "rlagent":    os.path.join(proc_base_dir, "processed_rlagent_data"),
            "simulation": os.path.join(proc_base_dir, "processed_simulation_data"),
        }
        for d in self.proc_dirs.values():
            os.makedirs(d, exist_ok=True)

    # --------------------------------------------------------
    # fit_scaler_from_file
    # --------------------------------------------------------

    def fit_scaler_from_file(self, symbol: str, method: str, feature_cols: list[str]):
        """
        Szuka pliku treningowego dla symbolu (substring) w processed_trainmodel_data,
        fituje scaler na feature_cols (bez EXCLUDE_COLUMNS_REALTIME i target).
        Zwraca (fitted_scaler, cols_fitted) lub None dla RAW/ROLLING_Z.
        """
        method = method.upper()
        if method not in PREPROCESSING_METHODS:
            raise ValueError(f"Unknown preprocessing method '{method}'. Allowed: {sorted(PREPROCESSING_METHODS)}")

        if method in ("RAW", "ROLLING_Z"):
            return None

        train_dir    = self.proc_dirs["trainmodel"]
        symbol_upper = symbol.upper()
        candidates   = sorted([
            f for f in os.listdir(train_dir)
            if symbol_upper in f.upper() and f.endswith(".csv")
        ])

        if not candidates:
            raise FileNotFoundError(
                f"No training file found for symbol '{symbol_upper}' in '{train_dir}'. "
                f"Available files: {os.listdir(train_dir)}"
            )

        df_train    = pd.read_csv(os.path.join(train_dir, candidates[0]))
        cols_to_fit = [
            c for c in feature_cols
            if c not in EXCLUDE_COLUMNS_REALTIME and c != "target" and c in df_train.columns
        ]

        if not cols_to_fit:
            raise ValueError(
                f"None of the requested feature_cols {feature_cols} found in '{candidates[0]}' "
                f"(after excluding EXCLUDE_COLUMNS_REALTIME and target)."
            )

        X_train = np.nan_to_num(df_train[cols_to_fit].values.astype(float), nan=0.0)
        scaler  = {"Z-SCORE": StandardScaler, "ROBUST": RobustScaler, "MINMAX": MinMaxScaler}[method]()
        scaler.fit(X_train)
        return scaler, cols_to_fit

    # --------------------------------------------------------
    # compute_features_for_realtime
    # --------------------------------------------------------

    @staticmethod
    def compute_features_for_realtime(
        df: pd.DataFrame,
        required_features: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Przyjmuje df z kolumnami OHLCV (bufor świec).
        Jeśli required_features=None — oblicza wszystkie cechy.
        Jeśli required_features podane — oblicza tylko potrzebne (+ ich dependencje).
        """
        needed = (
            _resolve_required(required_features)
            if required_features is not None
            else set(_FEATURE_DEPS.keys())
        )

        close  = df["close"]
        volume = df["volume"]
        high   = df["high"]
        low    = df["low"]

        # --- ret1 i lagi ---
        if "ret1" in needed:
            df["ret1"] = np.log(close).diff()
        for k in [1, 2, 5, 10]:
            if f"ret1_lag{k}" in needed:
                df[f"ret1_lag{k}"] = df["ret1"].shift(k)

        # --- rolling mean/std ---
        for w in [10, 50]:
            if f"roll_mean_{w}" in needed:
                df[f"roll_mean_{w}"] = close.rolling(w).mean()
            if f"roll_std_{w}" in needed:
                df[f"roll_std_{w}"] = close.rolling(w).std()

        # --- macd ---
        if "macd" in needed or "macd_sig" in needed:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            df["macd"] = ema12 - ema26
        if "macd_sig" in needed:
            df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()

        # --- set1 ---
        for w, col in [(5, "ma_5"), (10, "ma_10"), (20, "ma_20"), (10, "sma_10"), (30, "sma_30")]:
            if col in needed:
                df[col] = close.rolling(w).mean()
        if "cumvol" in needed:
            df["cumvol"] = volume.cumsum()
        if "volume_norm" in needed:
            df["volume_norm"] = (volume - volume.min()) / (volume.max() - volume.min() + 1e-9)
        if "std_20" in needed:
            df["std_20"] = close.rolling(20).std()

        # --- set2 ---
        if "RSI_14" in needed:
            delta = close.diff()
            gain  = delta.clip(lower=0)
            loss  = delta.clip(upper=0).abs()
            rs    = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
            df["RSI_14"] = 100 - (100 / (1 + rs))
        if "EMA_20" in needed:
            df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
        if "EMA_50" in needed:
            df["EMA_50"] = close.ewm(span=50, adjust=False).mean()

        # --- set3 ---
        if "BB_upper" in needed or "BB_lower" in needed:
            ma20  = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            if "BB_upper" in needed:
                df["BB_upper"] = ma20 + 2 * std20
            if "BB_lower" in needed:
                df["BB_lower"] = ma20 - 2 * std20
        if "Momentum_10" in needed:
            df["Momentum_10"] = close.diff(10)
        if "ROC_10" in needed:
            df["ROC_10"] = close.pct_change(10)

        # --- set4 ---
        if "ATR_14" in needed:
            close_arr  = close.values
            prev_close = np.roll(close_arr, 1); prev_close[0] = close_arr[0]
            tr = np.maximum(
                high.values - low.values,
                np.maximum(np.abs(high.values - prev_close), np.abs(low.values - prev_close))
            )
            df["ATR_14"] = pd.Series(tr, index=df.index).rolling(14).mean()
        if "VWAP" in needed:
            df["VWAP"] = (close * volume).cumsum() / (volume.cumsum() + 1e-9)
        if "ROC_5" in needed:
            df["ROC_5"] = close.pct_change(5)

        # --- target ---
        if "target" in needed:
            df["target"] = (close.diff() > 0).astype(int).shift(1).fillna(0).astype(int)

        df.fillna(0.0, inplace=True)
        return df

    # --------------------------------------------------------
    # apply_preprocessing_realtime
    # --------------------------------------------------------

    @staticmethod
    def _rolling_zscore_window(X: np.ndarray, min_periods: int = 20) -> np.ndarray:
        df = pd.DataFrame(X.astype(float))
        mu = df.expanding(min_periods=min_periods).mean()
        sd = df.expanding(min_periods=min_periods).std().replace(0, np.nan)
        return (df - mu).divide(sd).fillna(0.0).values

    @staticmethod
    def apply_preprocessing_realtime(
        df: pd.DataFrame,
        method: str,
        scaler,
        cols_to_scale: list[str],
    ) -> pd.DataFrame:
        """
        Transform feature kolumn przez fitted scaler.
        OHLCV i metadane (EXCLUDE_COLUMNS_REALTIME) są pomijane.
        """
        if method == "RAW":
            return df

        if method == "ROLLING_Z":
            cols = [c for c in cols_to_scale if c in df.columns and c != "target"]
            if not cols:
                cols = [
                    c for c in df.columns
                    if c not in EXCLUDE_COLUMNS_REALTIME
                    and c != "target"
                    and pd.api.types.is_numeric_dtype(df[c])
                ]
            if cols:
                df = df.copy()
                df[cols] = np.round(DataServicePlus._rolling_zscore_window(df[cols].values.astype(float)), 5)
            if "target" in df.columns:
                df["target"] = df["target"].astype(int)
            return df

        if scaler is None or not cols_to_scale:
            return df

        cols = [c for c in cols_to_scale if c in df.columns and c != "target"]
        if not cols:
            return df

        df = df.copy()
        df[cols] = np.round(scaler.transform(df[cols].values.astype(float)), 5)
        if "target" in df.columns:
            df["target"] = df["target"].astype(int)
        return df