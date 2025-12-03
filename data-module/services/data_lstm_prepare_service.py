import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List
import joblib
from datetime import datetime

class DataLSTMPrepareService:

    EXCLUDE_COLUMNS = [
        "open_time",
        "open",
        "high",
        "low",
        "close_time"
    ]

    def __init__(self, filename=None, seq_len=64, scaler_type="standard"):
        print("[LSTM] Initializing DataLSTMPrepareService...")
        self.filename = filename
        self.seq_len = seq_len
        self.scaler_type = scaler_type

        self.base_dir = os.getcwd()
        self.proc_base_dir = os.path.join(self.base_dir, "shared", "files")

        self.proc_dirs = {
            "trainmodel_lstm": os.path.join(self.proc_base_dir, "processed_trainmodel_data_lstm"),
            "testmodel_lstm": os.path.join(self.proc_base_dir, "processed_testmodel_data_lstm"),
            "rlagent_lstm": os.path.join(self.proc_base_dir, "processed_rlagent_data_lstm"),
            "simulation_lstm": os.path.join(self.proc_base_dir, "processed_simulation_data_lstm"),
        }

        for d in self.proc_dirs.values():
            os.makedirs(d, exist_ok=True)

        self.df = None
        self.scaler_features = None
        self.scaler_price = None
        print("[LSTM] Initialization complete.")

    def set_filename(self, filename: str):
        print(f"[LSTM] Setting filename to: {filename}")
        self.filename = filename

    def load_data(self):
        print("[LSTM] Loading data...")
        if not self.filename:
            raise ValueError("Filename is not set")

        path = os.path.join(self.base_dir, "shared", "files", "raw_data", self.filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File '{path}' not found!")

        self.df = pd.read_csv(path)
        print(f"[LSTM] Data loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
        return self

    # ---------------------------------
    # Technical Indicators Sets
    def add_technical_indicators(self, set_name: str = "set1"):
        if self.df is None:
            raise ValueError("Data not loaded")

        close = self.df["close"].values
        volume = self.df["volume"].values

        # --- set1 ---
        if set_name == "set1":
            self.df["ma_5"] = pd.Series(close).rolling(5).mean()
            self.df["ma_10"] = pd.Series(close).rolling(10).mean()
            self.df["ma_20"] = pd.Series(close).rolling(20).mean()
            self.df["sma_10"] = pd.Series(close).rolling(10).mean()
            self.df["sma_30"] = pd.Series(close).rolling(30).mean()
            self.df["cumvol"] = volume.cumsum()
            self.df["volume_norm"] = (volume - volume.min()) / (volume.max() - volume.min())
            self.df["std_20"] = pd.Series(close).rolling(20).std()

        # --- set2 ---
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

        # --- set3 ---
        elif set_name == "set3":
            ma20 = pd.Series(close).rolling(20).mean()
            std20 = pd.Series(close).rolling(20).std()
            self.df["BB_upper"] = ma20 + 2 * std20
            self.df["BB_lower"] = ma20 - 2 * std20
            self.df["Momentum_10"] = pd.Series(close).diff(10)
            self.df["ROC_10"] = pd.Series(close).pct_change(10)

        # --- set4 ---
        elif set_name == "set4":
            high = self.df["high"].values
            low = self.df["low"].values
            close_s = self.df["close"].values
            tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close_s, 1)), abs(low - np.roll(close_s, 1))))
            self.df["ATR_14"] = pd.Series(tr).rolling(14).mean()
            self.df["VWAP"] = (self.df["close"] * self.df["volume"]).cumsum() / (self.df["volume"].cumsum() + 1e-9)
            self.df["ROC_5"] = pd.Series(close).pct_change(5)

        # --- all sets ---
        elif set_name == "set_all":
            for s in ["set1", "set2", "set3", "set4"]:
                self.add_technical_indicators(s)

        else:
            raise ValueError(f"Unknown technical indicators set: {set_name}")

        self.df = self.df.dropna().reset_index(drop=True)
        return self

    # ---------------------------------
    def create_sequences(self, X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i + seq_len])
            ys.append(y[i + seq_len])
        return np.array(Xs), np.array(ys)

    def fit_scalers(self, X_train, y_train, scaler_dir):
        self.scaler_features = StandardScaler() if self.scaler_type == "standard" else MinMaxScaler()
        self.scaler_price = StandardScaler() if self.scaler_type == "standard" else MinMaxScaler()

        self.scaler_features.fit(X_train)
        self.scaler_price.fit(y_train.reshape(-1, 1))

        joblib.dump(self.scaler_features, os.path.join(scaler_dir, "scaler_features.pkl"))
        joblib.dump(self.scaler_price, os.path.join(scaler_dir, "scaler_price.pkl"))
        print(f"[LSTM] Scalery zapisane w {scaler_dir}")

    def scale_data(self, X, y):
        X_scaled = self.scaler_features.transform(X)
        y_scaled = self.scaler_price.transform(y.reshape(-1, 1)).flatten()
        return X_scaled, y_scaled

    # ---------------------------------
    def run_pipeline(
        self,
        steps: List[str],
        preprocessing_method: str = "RAW",
        output_folders: List[str] = None,
        technical_indicators_set: str = "set1"
    ):
        if "load_data" in steps:
            self.load_data()

        if self.df is None:
            raise RuntimeError("Dataframe is empty after load_data().")

        # --- Preprocessing ---
        df = self.df.copy()
        if preprocessing_method.upper() == "DIFF":
            df = df.diff().dropna()
        elif preprocessing_method.upper() == "LOG":
            df = np.log1p(df)
        elif preprocessing_method.upper() != "RAW":
            raise ValueError(f"Unknown preprocessing method: {preprocessing_method}")

        # --- Add technical indicators ---
        if technical_indicators_set:
            self.add_technical_indicators(technical_indicators_set)

        # --- Remove excluded columns and keep numeric ---
        df = df.drop(columns=[col for col in self.EXCLUDE_COLUMNS if col in df.columns], errors='ignore')
        df = df.select_dtypes(include=[np.number])
        if df.shape[1] < 2:
            raise RuntimeError("Not enough numeric columns for LSTM.")

        values = df.values
        X = values[:, :-1]
        y = values[:, -1]

        # --- Train/Test split ---
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        if not output_folders:
            raise ValueError("output_folders cannot be empty")

        # --- Tworzenie unikalnego folderu eksperymentu ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {}
        for key in output_folders:
            if key not in self.proc_dirs:
                raise ValueError(f"Invalid output folder: {key}")

            base_out_dir = self.proc_dirs[key]
            experiment_dir = os.path.join(base_out_dir, timestamp)
            os.makedirs(experiment_dir, exist_ok=True)

            # --- Fit scalers only on train ---
            self.fit_scalers(X_train, y_train, experiment_dir)

            # --- Scale ---
            X_train_scaled, y_train_scaled = self.scale_data(X_train, y_train)
            X_test_scaled, y_test_scaled = self.scale_data(X_test, y_test)

            # --- Create sequences ---
            Xtr, ytr = self.create_sequences(X_train_scaled, y_train_scaled, self.seq_len)
            Xte, yte = self.create_sequences(X_test_scaled, y_test_scaled, self.seq_len)

            # --- Save ---
            out_filename = f"processed_{preprocessing_method.upper()}_{technical_indicators_set}_{self.filename}_lstm"
            np.save(os.path.join(experiment_dir, f"X_train_{out_filename}.npy"), Xtr)
            np.save(os.path.join(experiment_dir, f"y_train_{out_filename}.npy"), ytr)
            np.save(os.path.join(experiment_dir, f"X_test_{out_filename}.npy"), Xte)
            np.save(os.path.join(experiment_dir, f"y_test_{out_filename}.npy"), yte)

            results[key] = {
                "folder": experiment_dir,
                "filename": out_filename,
                "X_train": f"X_train_{out_filename}.npy",
                "y_train": f"y_train_{out_filename}.npy",
                "X_test": f"X_test_{out_filename}.npy",
                "y_test": f"y_test_{out_filename}.npy",
                "scaler_features": os.path.join(experiment_dir, "scaler_features.pkl"),
                "scaler_price": os.path.join(experiment_dir, "scaler_price.pkl")
            }
            print(f"[LSTM] Folder '{experiment_dir}' processed successfully.")

        return results
