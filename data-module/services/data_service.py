import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

class DataService:

    BASE_OUTPUT_DIR = "shared/files/processed_trainmodel_data"
    EXCLUDE_COLUMNS = ["timestamp", "open_time", "close_time"]

    def __init__(self):
        self.base_dir = os.getcwd()
        self.raw_dir = os.path.join(self.base_dir, "shared", "files", "raw_data")
        self.df: pd.DataFrame | None = None
        self.filename = None
        self.horizon = None
        self.output_dir = None

    # -------------------- PIPELINE --------------------
    def load_data(self):
        path = os.path.join(self.raw_dir, self.filename)
        self.df = pd.read_csv(path)
        if "timestamp" in self.df.columns:
            self.df.rename(columns={"timestamp": "open_time"}, inplace=True)

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        return self

    def make_features(self):
        d = self.df.copy()
        d["ret1"] = np.log(d["close"]).diff()
        for k in [1, 2, 5, 10]:
            d[f"ret1_lag{k}"] = d["ret1"].shift(k)
        for w in [10, 50]:
            d[f"roll_mean_{w}"] = d["close"].rolling(w).mean()
            d[f"roll_std_{w}"] = d["close"].rolling(w).std()
        # MACD
        ema12 = d["close"].ewm(span=12, adjust=False).mean()
        ema26 = d["close"].ewm(span=26, adjust=False).mean()
        d["macd"] = ema12 - ema26
        d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()
        # target
        d["target"] = (d["close"].shift(-self.horizon) > d["close"]).astype(int)
        self.df = d.dropna().reset_index(drop=True)
        return self

    def add_technical_indicators(self):
        close = self.df["close"]
        self.df["ma_5"] = close.rolling(5).mean()
        self.df["ma_10"] = close.rolling(10).mean()
        self.df["ma_20"] = close.rolling(20).mean()
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
        self.df["RSI_14"] = 100 - (100 / (1 + rs))
        self.df["EMA_20"] = close.ewm(span=20).mean()
        self.df["EMA_50"] = close.ewm(span=50).mean()
        self.df = self.df.dropna().reset_index(drop=True)
        return self

    def apply_preprocessing(self, method: str):
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        numeric_cols = [c for c in numeric_cols if c not in self.EXCLUDE_COLUMNS]
        X = self.df[numeric_cols]

        if method == "RAW":
            pass
        elif method == "Z-SCORE":
            X = StandardScaler().fit_transform(X)
        elif method == "ROBUST":
            X = RobustScaler().fit_transform(X)
        elif method == "MINMAX":
            X = MinMaxScaler().fit_transform(X)
        else:
            raise ValueError(f"Unknown preprocessing: {method}")

        self.df[numeric_cols] = X
        return self

    # -------------------- PUBLIC --------------------
    def generate_and_save(
        self,
        filename: str,
        horizon: int,
        output_folder: str,
        steps: list[str],
        preprocessing: str,
        feature_combo: list[str],
    ) -> str:

        self.filename = filename
        self.horizon = horizon
        self.output_dir = os.path.join(self.base_dir, "shared/files", output_folder)
        os.makedirs(self.output_dir, exist_ok=True)

        step_map = {
            "load_data": self.load_data,
            "make_features": self.make_features,
            "add_technical_indicators": self.add_technical_indicators,
        }

        for step in steps:
            step_map[step]()

        self.apply_preprocessing(preprocessing)

        # -------------------- feature do modelu --------------------
        feature_cols = [c for c in feature_combo if c in self.df.columns]
        df_features = self.df[feature_cols + ["target"]].dropna(subset=['target']).reset_index(drop=True)

        # -------------------- bazowe kolumny --------------------
        base_cols = ["open_time","open","high","low","close","volume",
                     "close_time","quote_asset_volume","number_of_trades",
                     "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"]
        base_cols_existing = [c for c in base_cols if c in self.df.columns]
        df_base = self.df[base_cols_existing].reset_index(drop=True)

        # łączymy bazowe kolumny + feature + target
        df_out = pd.concat([df_base, df_features], axis=1)

        # -------------------- zapis --------------------
        feature_tag = "__".join(feature_cols)
        fname = f"{filename.replace('.csv','')}_h{horizon}_{feature_tag}.csv"
        out_path = os.path.join(self.output_dir, fname)
        df_out.to_csv(out_path, index=False)
        return out_path

    # -------------------- HELPER --------------------
    def get_model_features(self):
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        return [c for c in numeric_cols if c != "target"]
