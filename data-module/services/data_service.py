import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from services.operation_tracker import track_operation
from configuration.feature_sets_configuration import FEATURE_SETS


class DataService:

    BASE_OUTPUT_DIR = "processed_trainmodel_data"
    EXCLUDE_COLUMNS = ["timestamp", "open_time", "close_time"]

    def __init__(self):

        current_dir = Path(__file__).resolve().parent

        self.base_dir = current_dir.parent.parent

        self.raw_dir = self.base_dir / "shared" / "files" / "raw_data"

        self.df: pd.DataFrame | None = None
        self.filename = None
        self.horizon = None
        self.output_dir = None

    def load_data(self):

        path = self.raw_dir / self.filename

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

        ema12 = d["close"].ewm(span=12, adjust=False).mean()
        ema26 = d["close"].ewm(span=26, adjust=False).mean()

        d["macd"] = ema12 - ema26
        d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()

        self.df = d.dropna().reset_index(drop=True)

        return self

    def add_technical_indicators(self):

        close = self.df["close"]
        volume = self.df["volume"]
        high = self.df["high"]
        low = self.df["low"]

        # ===== istniejące =====

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

        # ===== set1 =====

        self.df["sma_10"] = close.rolling(10).mean()
        self.df["sma_30"] = close.rolling(30).mean()

        self.df["cumvol"] = volume.cumsum()

        self.df["volume_norm"] = (volume - volume.min()) / (volume.max() - volume.min())

        self.df["std_20"] = close.rolling(20).std()

        # ===== set3 =====

        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()

        self.df["BB_upper"] = ma20 + 2 * std20
        self.df["BB_lower"] = ma20 - 2 * std20

        self.df["Momentum_10"] = close.diff(10)
        self.df["ROC_10"] = close.pct_change(10)

        # ===== set4 =====

        tr = np.maximum(
            high - low,
            np.maximum(
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            )
        )

        self.df["ATR_14"] = pd.Series(tr).rolling(14).mean()

        self.df["VWAP"] = (close * volume).cumsum() / (volume.cumsum() + 1e-9)

        self.df["ROC_5"] = close.pct_change(5)

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
        
        elif method == "ROLLING_Z":
            rolling_mean = X.rolling(window=20, min_periods=1).mean()
            rolling_std = X.rolling(window=20, min_periods=1).std()
            X = (X - rolling_mean) / (rolling_std + 1e-9)

        else:
            raise ValueError(f"Unknown preprocessing: {method}")

        self.df[numeric_cols] = X

        return self

    @track_operation("data_service", "generate_and_save", "data_processing")
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

        folder_name = Path(output_folder).name

        self.output_dir = self.base_dir / "shared" / "files" / folder_name

        self.output_dir.mkdir(parents=True, exist_ok=True)

        step_map = {
            "load_data": self.load_data,
            "make_features": self.make_features,
            "add_technical_indicators": self.add_technical_indicators,
        }

        for step in steps:
            step_map[step]()

        self.apply_preprocessing(preprocessing)

        expanded_features = []

        for f in feature_combo:

            if f in FEATURE_SETS:
                expanded_features.extend(FEATURE_SETS[f])
            else:
                expanded_features.append(f)

        feature_cols = [c for c in expanded_features if c in self.df.columns]

        df_features = self.df[feature_cols].reset_index(drop=True)

        base_cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ]

        base_cols_existing = [c for c in base_cols if c in self.df.columns]

        df_base = self.df[base_cols_existing].reset_index(drop=True)

        df_out = pd.concat([df_base, df_features], axis=1)

        feature_tag = "__".join(feature_cols)

        fname = f"{filename.replace('.csv','')}_h{horizon}_{feature_tag}.csv"

        out_path = self.output_dir / fname

        df_out.to_csv(out_path, index=False)
        print(out_path.as_posix())
        return out_path.as_posix()

    def get_model_features(self):

        numeric_cols = self.df.select_dtypes(include=np.number).columns

        return list(numeric_cols)