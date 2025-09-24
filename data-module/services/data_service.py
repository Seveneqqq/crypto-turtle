import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

class RollingZScore(BaseEstimator, TransformerMixin):
    def __init__(self, min_periods=50):
        self.min_periods = min_periods

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).astype(float)
        mu = X.expanding(min_periods=self.min_periods).mean()
        sd = X.expanding(min_periods=self.min_periods).std().replace(0, np.nan)
        Z = (X - mu) / sd
        return Z.fillna(0.0).values

def transform_rolling_no_leak(Xtr_raw, Xte_raw):
    rz = RollingZScore()
    Xtr = rz.fit_transform(Xtr_raw)
    concat = np.vstack([Xtr_raw, Xte_raw])
    Xall = RollingZScore().fit_transform(concat)
    Xte = Xall[len(Xtr_raw):]
    return Xtr, Xte

class DataService:
    def __init__(self, filename: str = None, horizon: int = 15):
        self.filename = filename
        self.horizon = horizon
        self.base_dir = os.getcwd()
        self.raw_dir = os.path.join(self.base_dir, "shared", "files", "binance_historical_raw_data")
        self.proc_dir = os.path.join(self.base_dir, "shared", "files", "processed_files")
        os.makedirs(self.proc_dir, exist_ok=True)
        self.df = None

    def set_filename(self, filename: str):
        """Ustawia plik CSV, który ma zostać przetworzony."""
        self.filename = filename

    def load_data(self):
        if not self.filename:
            raise ValueError("Filename is not set")
        path = os.path.join(self.raw_dir, self.filename)
        self.df = pd.read_csv(path)
        return self

    def make_features(self):
        if self.df is None:
            raise ValueError("Data not loaded")
        d = self.df.copy()
        d["ret1"] = np.log(d["close"]).diff()
        for k in [1, 2, 5, 10]:
            d[f"ret1_lag{k}"] = d["ret1"].shift(k)
        for w in [10, 50]:
            d[f"roll_mean_{w}"] = d["close"].rolling(w).mean()
            d[f"roll_std_{w}"] = d["close"].rolling(w).std()
        ema12 = d["close"].ewm(span=12, adjust=False).mean()
        ema26 = d["close"].ewm(span=26, adjust=False).mean()
        d["macd"] = ema12 - ema26
        d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()
        d["target"] = (d["close"].shift(-self.horizon) > d["close"]).astype(int)
        self.df = d.dropna().reset_index(drop=True)
        return self

    def add_technical_indicators(self):
        if self.df is None:
            raise ValueError("Data not loaded")
        close = self.df["close"].values
        volume = self.df["volume"].values
        self.df["ma_5"] = pd.Series(close).rolling(5).mean()
        self.df["ma_10"] = pd.Series(close).rolling(10).mean()
        self.df["ma_20"] = pd.Series(close).rolling(20).mean()
        self.df["sma_10"] = pd.Series(close).rolling(10).mean()
        self.df["sma_30"] = pd.Series(close).rolling(30).mean()
        self.df["cumvol"] = volume.cumsum()
        self.df["volume_norm"] = (volume - volume.min()) / (volume.max() - volume.min())
        self.df["std_20"] = pd.Series(close).rolling(20).std()
        return self

    def apply_preprocessing(self, methods: list[str]):
        if self.df is None:
            raise ValueError("Data not loaded")
        
        X = self.df.select_dtypes(include=np.number).drop(columns=["target"], errors="ignore").values
        X_current = X.copy()
        applied_methods = []

        for method in methods:
            method_upper = method.upper()
            if method_upper == "RAW":
                pass
            elif method_upper == "Z-SCORE":
                sc = StandardScaler()
                X_current = sc.fit_transform(X_current)
            elif method_upper == "ROBUST":
                sc = RobustScaler()
                X_current = sc.fit_transform(X_current)
            elif method_upper == "MINMAX":
                sc = MinMaxScaler()
                X_current = sc.fit_transform(X_current)
            elif method_upper == "ROLLING_Z":
                _, X_current = transform_rolling_no_leak(X, X_current)
            else:
                raise ValueError(f"Unknown preprocessing method: {method}")
            applied_methods.append(method_upper)
        
        return X_current, applied_methods

    def run_pipeline(self, steps: list[str], preprocessing_methods: list[str] = ["RAW"]):
        if not self.filename:
            raise ValueError("Filename is not set")
        
        step_map = {
            "load_data": self.load_data,
            "make_features": self.make_features,
            "add_technical_indicators": self.add_technical_indicators,
        }
        for step in steps:
            if step in step_map:
                step_map[step]()

        X_processed, applied_methods = self.apply_preprocessing(preprocessing_methods)

        out_filename = f"processed_{'_'.join(applied_methods)}_{self.filename}"
        out_path = os.path.join(self.proc_dir, out_filename)
        
        df_out = self.df.copy()
        numeric_cols = df_out.select_dtypes(include=np.number).drop(columns=["target"], errors="ignore").columns
        df_out[numeric_cols] = X_processed
        df_out.to_csv(out_path, index=False)

        return {
            "filename": out_filename,
            "filepath": out_path,
            "applied_methods": applied_methods
        }
