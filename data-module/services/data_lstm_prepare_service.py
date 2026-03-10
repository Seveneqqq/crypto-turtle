import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict
from pathlib import Path

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

from services.operation_tracker import track_operation


class DataLSTMPrepareService:

    EXCLUDE_COLUMNS = [
        "open_time",
        "open",
        "high",
        "low",
        "close_time",
    ]

    def __init__(self, filename: str | None = None, seq_len: int = 64, scaler_type: str = "standard"):

        print("[LSTM] Initializing DataLSTMPrepareService")

        self.filename = filename
        self.seq_len = seq_len
        self.scaler_type = scaler_type

        # ROOT PROJEKTU (działa w Dockerze i lokalnie)
        self.project_root = Path(__file__).resolve().parents[2]

        # shared/files
        self.shared_dir = self.project_root / "shared" / "files"

        # katalog raw data
        self.raw_dir = self.shared_dir / "raw_data"

        # katalogi wynikowe
        self.proc_dirs = {
            "trainmodel_lstm": self.shared_dir / "processed_trainmodel_data_lstm",
            "testmodel_lstm": self.shared_dir / "processed_testmodel_data_lstm",
            "rlagent_lstm": self.shared_dir / "processed_rlagent_data_lstm",
            "simulation_lstm": self.shared_dir / "processed_simulation_data_lstm",
        }

        # tworzenie katalogów
        for d in self.proc_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        self.df: pd.DataFrame | None = None
        self.scaler_features = None
        self.scaler_price = None

        print(f"[LSTM] Project root: {self.project_root}")
        print("[LSTM] Initialization complete")

    def set_filename(self, filename: str):
        print(f"[LSTM] Setting filename to: {filename}")
        self.filename = filename

    def load_data(self):

        if not self.filename:
            raise ValueError("Filename is not set")

        path = self.raw_dir / self.filename

        if not path.exists():
            raise FileNotFoundError(f"File '{path}' not found")

        self.df = pd.read_csv(path)

        print(f"[LSTM] Data loaded: {len(self.df)} rows, {len(self.df.columns)} columns")

        return self

    def add_technical_indicators(self, set_name: str = "set1"):

        if self.df is None:
            raise ValueError("Data not loaded")

        df_proc = self.df.drop(columns=[col for col in self.EXCLUDE_COLUMNS if col in self.df.columns], errors="ignore")

        close = df_proc["close"].values
        volume = df_proc["volume"].values if "volume" in df_proc.columns else None

        if set_name == "set1" and volume is not None:

            df_proc["ma_5"] = pd.Series(close).rolling(5).mean()
            df_proc["ma_10"] = pd.Series(close).rolling(10).mean()
            df_proc["ma_20"] = pd.Series(close).rolling(20).mean()

            df_proc["sma_10"] = pd.Series(close).rolling(10).mean()
            df_proc["sma_30"] = pd.Series(close).rolling(30).mean()

            df_proc["cumvol"] = volume.cumsum()

            df_proc["volume_norm"] = (volume - volume.min()) / (volume.max() - volume.min() + 1e-9)

            df_proc["std_20"] = pd.Series(close).rolling(20).std()

        elif set_name == "set2":

            delta = pd.Series(close).diff()

            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)

            avg_gain = pd.Series(gain).rolling(14).mean()
            avg_loss = pd.Series(loss).rolling(14).mean()

            rs = avg_gain / (avg_loss + 1e-9)

            df_proc["RSI_14"] = 100 - (100 / (1 + rs))

            df_proc["EMA_20"] = pd.Series(close).ewm(span=20, adjust=False).mean()
            df_proc["EMA_50"] = pd.Series(close).ewm(span=50, adjust=False).mean()

        elif set_name == "set3":

            ma20 = pd.Series(close).rolling(20).mean()
            std20 = pd.Series(close).rolling(20).std()

            df_proc["BB_upper"] = ma20 + 2 * std20
            df_proc["BB_lower"] = ma20 - 2 * std20

            df_proc["Momentum_10"] = pd.Series(close).diff(10)
            df_proc["ROC_10"] = pd.Series(close).pct_change(10)

        elif set_name == "set4":

            high = df_proc["high"].values if "high" in df_proc.columns else None
            low = df_proc["low"].values if "low" in df_proc.columns else None

            if high is not None and low is not None:

                tr = np.maximum(
                    high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
                )

                df_proc["ATR_14"] = pd.Series(tr).rolling(14).mean()

                df_proc["VWAP"] = (df_proc["close"] * df_proc["volume"]).cumsum() / (df_proc["volume"].cumsum() + 1e-9)

                df_proc["ROC_5"] = pd.Series(close).pct_change(5)

        elif set_name == "set_all":

            for s in ["set1", "set2", "set3", "set4"]:
                self.add_technical_indicators(s)

            return self

        else:
            raise ValueError(f"Unknown technical indicators set: {set_name}")

        for col in df_proc.columns:
            self.df[col] = df_proc[col]

        self.df = self.df.dropna().reset_index(drop=True)

        return self

    def create_sequences(self, X, y, seq_len: int):

        Xs = []
        ys = []

        for i in range(len(X) - seq_len):
            Xs.append(X[i:i + seq_len])
            ys.append(y[i + seq_len])

        return np.array(Xs), np.array(ys)

    def fit_scalers(self, X_train, y_train, scaler_dir: Path):

        X_train = np.nan_to_num(X_train)
        y_train = np.nan_to_num(y_train)

        if self.scaler_type == "standard":
            self.scaler_features = StandardScaler()
            self.scaler_price = StandardScaler()
        else:
            self.scaler_features = MinMaxScaler()
            self.scaler_price = MinMaxScaler()

        self.scaler_features.fit(X_train)
        self.scaler_price.fit(y_train.reshape(-1, 1))

        joblib.dump(self.scaler_features, scaler_dir / "scaler_features.pkl")
        joblib.dump(self.scaler_price, scaler_dir / "scaler_price.pkl")

        print(f"[LSTM] Scalery zapisane w {scaler_dir}")

    def scale_data(self, X, y):

        X = np.nan_to_num(X)
        y = np.nan_to_num(y)

        X_scaled = self.scaler_features.transform(X)
        y_scaled = self.scaler_price.transform(y.reshape(-1, 1)).flatten()

        return X_scaled, y_scaled

    @track_operation("data_lstm_prepare_service", "run_pipeline", "lstm_data_processing")
    def run_pipeline(
        self,
        steps: List[str],
        preprocessing_method: str = "RAW",
        output_folders: List[str] | None = None,
        technical_indicators_set: str | None = "set1",
        selected_features: List[str] | None = None,
    ):

        if "load_data" in steps:
            self.load_data()

        if self.df is None:
            raise RuntimeError("Dataframe is empty after load_data()")

        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.fillna(0, inplace=True)

        if selected_features is not None:

            tech_cols = {
                "ma_5","ma_10","ma_20","sma_10","sma_30","cumvol","volume_norm",
                "std_20","RSI_14","EMA_20","EMA_50","BB_upper","BB_lower",
                "Momentum_10","ROC_10","ATR_14","VWAP","ROC_5"
            }

            if any(f in tech_cols for f in selected_features):
                self.add_technical_indicators("set_all")

            available_features = [f for f in selected_features if f in self.df.columns]

            if len(available_features) == 0:
                raise ValueError("No valid features after adding indicators")

        else:

            available_features = [c for c in self.df.columns if c not in self.EXCLUDE_COLUMNS]

        df = self.df.copy()

        method_upper = preprocessing_method.upper()

        if method_upper == "DIFF":

            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].diff()
            df = df.dropna()

        elif method_upper == "LOG":

            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = np.log1p(df[num_cols])

        elif method_upper != "RAW":
            raise ValueError(f"Unknown preprocessing method: {preprocessing_method}")

        X_all = df[available_features].values
        y_all = df["close"].values

        train_size = int(len(X_all) * 0.8)

        X_train = X_all[:train_size]
        X_test = X_all[train_size:]

        y_train = y_all[:train_size]
        y_test = y_all[train_size:]

        if not output_folders:
            raise ValueError("output_folders cannot be empty")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results: Dict[str, Dict[str, str]] = {}

        for key in output_folders:

            if key not in self.proc_dirs:
                raise ValueError(f"Invalid output folder: {key}")

            base_out_dir = self.proc_dirs[key]

            exp_dir = base_out_dir / timestamp
            exp_dir.mkdir(parents=True, exist_ok=True)

            self.fit_scalers(X_train, y_train, exp_dir)

            X_train_scaled, y_train_scaled = self.scale_data(X_train, y_train)
            X_test_scaled, y_test_scaled = self.scale_data(X_test, y_test)

            Xtr, ytr = self.create_sequences(X_train_scaled, y_train_scaled, self.seq_len)
            Xte, yte = self.create_sequences(X_test_scaled, y_test_scaled, self.seq_len)

            out_filename = f"processed_{method_upper}_{technical_indicators_set}_{self.filename}_lstm"

            np.save(exp_dir / f"X_train_{out_filename}.npy", Xtr)
            np.save(exp_dir / f"y_train_{out_filename}.npy", ytr)
            np.save(exp_dir / f"X_test_{out_filename}.npy", Xte)
            np.save(exp_dir / f"y_test_{out_filename}.npy", yte)

            results[key] = {
                "folder": str(exp_dir),
                "filename": out_filename,
                "X_train": f"X_train_{out_filename}.npy",
                "y_train": f"y_train_{out_filename}.npy",
                "X_test": f"X_test_{out_filename}.npy",
                "y_test": f"y_test_{out_filename}.npy",
                "scaler_features": str(exp_dir / "scaler_features.pkl"),
                "scaler_price": str(exp_dir / "scaler_price.pkl"),
            }

            print(f"[LSTM] Folder '{exp_dir}' processed successfully")

        return results