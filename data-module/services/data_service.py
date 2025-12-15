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
    EXCLUDE_COLUMNS = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "close_time",
        "target"
    ]

    def __init__(self, filename: str = None, horizon: int = 15):
        self.filename = filename
        self.horizon = horizon
        self.base_dir = os.getcwd()
        self.raw_dir = os.path.join(self.base_dir, "shared", "files", "raw_data")
        self.proc_base_dir = os.path.join(self.base_dir, "shared", "files")
        self.proc_dirs = {
            "trainmodel": os.path.join(self.proc_base_dir, "processed_trainmodel_data"),
            "testmodel": os.path.join(self.proc_base_dir, "processed_testmodel_data"),
            "rlagent": os.path.join(self.proc_base_dir, "processed_rlagent_data"),
            "simulation": os.path.join(self.proc_base_dir, "processed_simulation_data"),
        }
        for d in self.proc_dirs.values():
            os.makedirs(d, exist_ok=True)

        self.df = None
        self.technical_indicators_set = "set1"

    def set_filename(self, filename: str):
        self.filename = filename

    def load_data(self):
        if not self.filename:
            raise ValueError("Filename is not set")
        path = os.path.join(self.raw_dir, self.filename)
        self.df = pd.read_csv(path)

        if "timestamp" in self.df.columns:
            self.df.rename(columns={"timestamp": "open_time"}, inplace=True)

        numeric_cols = [
            "open", "high", "low", "close", "volume",
            "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
        ]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

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

    def add_technical_indicators(self, set_name: str = "set1"):
        if self.df is None:
            raise ValueError("Data not loaded")

        close = self.df["close"].values
        volume = self.df["volume"].values

        if set_name == "set_all":
            self.add_technical_indicators("set1")
            self.add_technical_indicators("set2")
            self.add_technical_indicators("set3")
            self.add_technical_indicators("set4")
            return self

        if set_name.startswith("set_") and set_name not in ["set1", "set2", "set3", "set4"]:
            feat = set_name.replace("set_", "")

            all_feats = {
                "ma_5", "ma_10", "ma_20", "sma_10", "sma_30",
                "cumvol", "volume_norm", "std_20",
                "RSI_14", "EMA_20", "EMA_50",
                "BB_upper", "BB_lower", "Momentum_10", "ROC_10",
                "ATR_14", "VWAP", "ROC_5",
            }

            if feat not in all_feats:
                raise ValueError(f"Feature '{feat}' nie istnieje")

            self.add_technical_indicators("set1")
            self.add_technical_indicators("set2")
            self.add_technical_indicators("set3")
            self.add_technical_indicators("set4")

            for col in list(self.df.columns):
                if col in all_feats and col != feat:
                    del self.df[col]

            return self


        if set_name == "set1":
            self.df["ma_5"] = pd.Series(close).rolling(5).mean()
            self.df["ma_10"] = pd.Series(close).rolling(10).mean()
            self.df["ma_20"] = pd.Series(close).rolling(20).mean()
            self.df["sma_10"] = pd.Series(close).rolling(10).mean()
            self.df["sma_30"] = pd.Series(close).rolling(30).mean()
            self.df["cumvol"] = volume.cumsum()
            self.df["volume_norm"] = (volume - volume.min()) / (volume.max() - volume.min())
            self.df["std_20"] = pd.Series(close).rolling(20).std()

        elif set_name == "set2":
            delta = pd.Series(close).diff()
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).rolling(14).mean()
            avg_loss = pd.Series(loss).rolling(14).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            self.df["RSI_14"] = 100 - (100 / (1 + rs))
            self.df["EMA_20"] = pd.Series(close).ewm(span=20, adjust=False).mean()
            self.df["EMA_50"] = pd.Series(close).ewm(span=50, adjust=False).mean()

        elif set_name == "set3":
            ma20 = pd.Series(close).rolling(20).mean()
            std20 = pd.Series(close).rolling(20).std()
            self.df["BB_upper"] = ma20 + 2 * std20
            self.df["BB_lower"] = ma20 - 2 * std20
            self.df["Momentum_10"] = pd.Series(close).diff(10)
            self.df["ROC_10"] = pd.Series(close).pct_change(10)

        elif set_name == "set4":
            high = self.df["high"].values
            low = self.df["low"].values
            close_s = self.df["close"].values
            tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close_s, 1)), abs(low - np.roll(close_s, 1))))
            self.df["ATR_14"] = pd.Series(tr).rolling(14).mean()
            self.df["VWAP"] = (self.df["close"] * self.df["volume"]).cumsum() / (self.df["volume"].cumsum() + 1e-9)
            self.df["ROC_5"] = pd.Series(close).pct_change(5)

        else:
            raise ValueError(f"Unknown technical indicators set: {set_name}")

        return self

    def apply_preprocessing(self, method: str):
        if self.df is None:
            raise ValueError("Data not loaded")

        numeric_cols = self.df.select_dtypes(include=np.number).columns
        numeric_cols = [c for c in numeric_cols if c not in self.EXCLUDE_COLUMNS]

        if len(numeric_cols) == 0:
            return np.array([]), method.upper(), []

        X = self.df[numeric_cols].values
        X_current = X.copy()

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

        return X_current, method_upper, numeric_cols

    def run_pipeline(
        self,
        steps: list[str],
        preprocessing_method: str = "RAW",
        output_folders: list[str] = None,
        technical_indicators_set: str = "set1"
    ):
        if not self.filename:
            raise ValueError("Filename is not set")

        self.technical_indicators_set = technical_indicators_set

        step_map = {
            "load_data": self.load_data,
            "make_features": self.make_features,
            "add_technical_indicators": lambda: self.add_technical_indicators(self.technical_indicators_set),
        }

        for step in steps:
            if step in step_map:
                step_map[step]()

        X_processed, applied_method, numeric_cols = self.apply_preprocessing(preprocessing_method)

        out_filename = f"processed_{applied_method}_{technical_indicators_set}_{self.filename}"
        df_out = self.df.copy()
        df_out[numeric_cols] = X_processed
        df_out[numeric_cols] = df_out[numeric_cols].round(5)

        results = {}
        if output_folders:
            for folder_key in output_folders:
                if folder_key in self.proc_dirs:
                    out_path = os.path.join(self.proc_dirs[folder_key], out_filename)
                    df_out.to_csv(out_path, index=False)
                    results[folder_key] = {
                        "filename": out_filename,
                        "filepath": out_path,
                        "applied_method": applied_method,
                        "technical_set": technical_indicators_set
                    }
        else:
            default_path = os.path.join(self.proc_dirs["trainmodel"], out_filename)
            df_out.to_csv(default_path, index=False)
            results["trainmodel"] = {
                "filename": out_filename,
                "filepath": default_path,
                "applied_method": applied_method,
                "technical_set": technical_indicators_set
            }

        return results
