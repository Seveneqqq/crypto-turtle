import os
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib


class DataLSTMPrepareService:
    """
    Przygotowanie danych pod LSTM:

    - load_data() z raw_data
    - dodanie wskaźników technicznych (set1–set4)
    - preprocessing: RAW / DIFF / LOG (na całym df)
    - usunięcie EXCLUDE_COLUMNS + nienumerycznych
    - target = kolumna 'close'
      sekwencje tworzymy tak, że y = close[i + seq_len]
    - split train/test (80/20, bez shuffle – dane czasowe)
    - dopasowanie skalerów TYLKO na train
    - stworzenie sekwencji 3D (samples, seq_len, features)
    - zapis .npy + scalerów .pkl
    """

    EXCLUDE_COLUMNS = [
        "open_time",
        "open",
        "high",
        "low",
        "close_time",
    ]

    def __init__(
        self,
        filename: str | None = None,
        seq_len: int = 64,
        scaler_type: str = "standard",
    ):
        print("[LSTM] Initializing DataLSTMPrepareService...")
        self.filename = filename
        self.seq_len = seq_len
        self.scaler_type = scaler_type

        self.base_dir = os.getcwd()
        self.proc_base_dir = os.path.join(self.base_dir, "shared", "files")

        self.proc_dirs: Dict[str, str] = {
            "trainmodel_lstm": os.path.join(
                self.proc_base_dir,
                "processed_trainmodel_data_lstm",
            ),
            "testmodel_lstm": os.path.join(
                self.proc_base_dir,
                "processed_testmodel_data_lstm",
            ),
            "rlagent_lstm": os.path.join(
                self.proc_base_dir,
                "processed_rlagent_data_lstm",
            ),
            "simulation_lstm": os.path.join(
                self.proc_base_dir,
                "processed_simulation_data_lstm",
            ),
        }

        for d in self.proc_dirs.values():
            os.makedirs(d, exist_ok=True)

        self.df: pd.DataFrame | None = None
        self.scaler_features = None
        self.scaler_price = None
        print("[LSTM] Initialization complete.")

    # ------------------------------------------------------------------
    # Ustawianie pliku i ładowanie danych
    # ------------------------------------------------------------------
    def set_filename(self, filename: str):
        print(f"[LSTM] Setting filename to: {filename}")
        self.filename = filename

    def load_data(self):
        print("[LSTM] Loading data...")
        if not self.filename:
            raise ValueError("Filename is not set")

        path = os.path.join(
            self.base_dir,
            "shared",
            "files",
            "raw_data",
            self.filename,
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"File '{path}' not found!")

        self.df = pd.read_csv(path)
        print(
            f"[LSTM] Data loaded: {len(self.df)} rows, {len(self.df.columns)} columns",
        )
        return self

    # ------------------------------------------------------------------
    # Wskaźniki techniczne – modyfikują self.df
    # ------------------------------------------------------------------
    def add_technical_indicators(self, set_name: str = "set1"):
        if self.df is None:
            raise ValueError("Data not loaded")

        if "close" not in self.df.columns or "volume" not in self.df.columns:
            raise ValueError(
                "Columns 'close' and 'volume' are required to compute indicators.",
            )

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
            self.df["volume_norm"] = (volume - volume.min()) / (
                volume.max() - volume.min() + 1e-9
            )
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
            self.df["EMA_20"] = pd.Series(close).ewm(
                span=20,
                adjust=False,
            ).mean()
            self.df["EMA_50"] = pd.Series(close).ewm(
                span=50,
                adjust=False,
            ).mean()

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
            if "high" not in self.df.columns or "low" not in self.df.columns:
                raise ValueError(
                    "Columns 'high' and 'low' are required for set4.",
                )
            high = self.df["high"].values
            low = self.df["low"].values
            close_s = self.df["close"].values
            tr = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - np.roll(close_s, 1)),
                    np.abs(low - np.roll(close_s, 1)),
                ),
            )
            self.df["ATR_14"] = pd.Series(tr).rolling(14).mean()
            self.df["VWAP"] = (
                self.df["close"] * self.df["volume"]
            ).cumsum() / (self.df["volume"].cumsum() + 1e-9)
            self.df["ROC_5"] = pd.Series(close).pct_change(5)

        # --- all sets ---
        elif set_name == "set_all":
            for s in ["set1", "set2", "set3", "set4"]:
                self.add_technical_indicators(s)

        else:
            raise ValueError(f"Unknown technical indicators set: {set_name}")

        # usuwamy wiersze z NaN po rollingach
        self.df = self.df.dropna().reset_index(drop=True)
        return self

    # ------------------------------------------------------------------
    # Sekwencje & skalowanie
    # ------------------------------------------------------------------
    def create_sequences(self, X, y, seq_len: int):
        """
        Tworzy sekwencje 3D pod LSTM.

        WAŻNE: zakładamy, że y jest serią 'close' (po skalowaniu).
        Sekwencje są zdefiniowane jako:
          - X_seq[i]  = X[i : i+seq_len]
          - y_seq[i]  = y[i + seq_len]

        Czyli:
          y_seq[i] = close[i + seq_len]
        """
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i : i + seq_len])
            ys.append(y[i + seq_len])
        return np.array(Xs), np.array(ys)

    def fit_scalers(self, X_train, y_train, scaler_dir: str):
        if self.scaler_type == "standard":
            self.scaler_features = StandardScaler()
            self.scaler_price = StandardScaler()
        else:
            self.scaler_features = MinMaxScaler()
            self.scaler_price = MinMaxScaler()

        self.scaler_features.fit(X_train)
        self.scaler_price.fit(y_train.reshape(-1, 1))

        joblib.dump(
            self.scaler_features,
            os.path.join(scaler_dir, "scaler_features.pkl"),
        )
        joblib.dump(
            self.scaler_price,
            os.path.join(scaler_dir, "scaler_price.pkl"),
        )
        print(f"[LSTM] Scalery zapisane w {scaler_dir}")

    def scale_data(self, X, y):
        X_scaled = self.scaler_features.transform(X)
        y_scaled = self.scaler_price.transform(y.reshape(-1, 1)).flatten()
        return X_scaled, y_scaled

    # ------------------------------------------------------------------
    # GŁÓWNY PIPELINE
    # ------------------------------------------------------------------
    def run_pipeline(
        self,
        steps: List[str],
        preprocessing_method: str = "RAW",
        output_folders: List[str] | None = None,
        technical_indicators_set: str | None = "set1",
    ):
        """
        Kroki:
        - (opcjonalnie) load_data
        - (opcjonalnie) add_technical_indicators na CAŁYM zbiorze
        - preprocessing (RAW / DIFF / LOG)
        - drop EXCLUDE_COLUMNS + tylko numeryczne
        - X: wszystkie cechy numeryczne
        - y: kolumna 'close'
        - sekwencje: y_seq[i] = close[i + seq_len]
        - split train/test (80/20, bez shuffle)
        - scalery na train, zapis .pkl
        - sekwencje 3D, zapis .npy
        """

        # 1) Ładowanie danych
        if "load_data" in steps:
            self.load_data()

        if self.df is None:
            raise RuntimeError("Dataframe is empty after load_data().")

        # 2) Dodanie wskaźników technicznych na CAŁYM zbiorze
        if technical_indicators_set:
            self.add_technical_indicators(technical_indicators_set)

        # 3) Kopia robocza df (po wskaźnikach)
        df = self.df.copy()

        # 4) Preprocessing
        method_upper = preprocessing_method.upper()
        if method_upper == "DIFF":
            # różnicujemy tylko kolumny numeryczne
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].diff()
            df = df.dropna()
        elif method_upper == "LOG":
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = np.log1p(df[num_cols])
        elif method_upper != "RAW":
            raise ValueError(
                f"Unknown preprocessing method: {preprocessing_method}",
            )

        # 5) Usunięcie niepotrzebnych kolumn + tylko numeryczne
        df = df.drop(
            columns=[col for col in self.EXCLUDE_COLUMNS if col in df.columns],
            errors="ignore",
        )
        df = df.select_dtypes(include=[np.number])

        if "close" not in df.columns:
            raise RuntimeError(
                "Column 'close' not found after preprocessing – cannot build target.",
            )

        if df.shape[0] <= self.seq_len + 1:
            raise RuntimeError(
                f"Not enough rows ({df.shape[0]}) for seq_len={self.seq_len}.",
            )

        # 6) X / y – target = 'close'
        #    UWAGA: X zawiera również kolumnę 'close' (przeszłe wartości),
        #    a y dla sekwencji będzie close[i + seq_len].
        y_all = df["close"].values
        X_all = df.values

        # 7) Train/Test split (80/20, bez shuffle – dane czasowe)
        train_size = int(len(X_all) * 0.8)
        X_train, X_test = X_all[:train_size], X_all[train_size:]
        y_train, y_test = y_all[:train_size], y_all[train_size:]

        if not output_folders:
            raise ValueError("output_folders cannot be empty")

        # 😎 Unikalny folder eksperymentu
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results: Dict[str, Dict[str, str]] = {}
        for key in output_folders:
            if key not in self.proc_dirs:
                raise ValueError(f"Invalid output folder: {key}")

            base_out_dir = self.proc_dirs[key]
            experiment_dir = os.path.join(base_out_dir, timestamp)
            os.makedirs(experiment_dir, exist_ok=True)

            # 9) Fit scalers tylko na train
            self.fit_scalers(X_train, y_train, experiment_dir)

            # 10) Skalowanie
            X_train_scaled, y_train_scaled = self.scale_data(X_train, y_train)
            X_test_scaled, y_test_scaled = self.scale_data(X_test, y_test)

            if len(X_train_scaled) <= self.seq_len or len(X_test_scaled) <= self.seq_len:
                raise RuntimeError(
                    f"Not enough samples after split for seq_len={self.seq_len}.",
                )

            # 11) Sekwencje
            # y_seq[i] = y_scaled[i + seq_len] => target = close[i+seq_len]
            Xtr, ytr = self.create_sequences(
                X_train_scaled,
                y_train_scaled,
                self.seq_len,
            )
            Xte, yte = self.create_sequences(
                X_test_scaled,
                y_test_scaled,
                self.seq_len,
            )

            # 12) Zapis .npy
            out_filename = (
                f"processed_{method_upper}_"
                f"{technical_indicators_set}_{self.filename}_lstm"
            )

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
                "scaler_features": os.path.join(
                    experiment_dir,
                    "scaler_features.pkl",
                ),
                "scaler_price": os.path.join(
                    experiment_dir,
                    "scaler_price.pkl",
                ),
                "preprocessing": method_upper,
                "technical_indicators_set": technical_indicators_set,
            }

            print(f"[LSTM] Folder '{experiment_dir}' processed successfully.")

        return results