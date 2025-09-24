import os
import pandas as pd
from datetime import datetime
import importlib


class SklearnDataPrepService:
    def __init__(self, shared_dir: str = None):
        # katalog /shared
        if shared_dir is None:
            shared_dir = self._find_project_root()
        self.PROCESSED_DIR = os.path.join(shared_dir, "shared", "files", "processed_files")
        os.makedirs(self.PROCESSED_DIR, exist_ok=True)
        print(f"[SklearnDataPrepService] Pliki będą zapisywane w: {self.PROCESSED_DIR}")

    @staticmethod
    def _find_project_root():
        # szuka katalogu głównego (tam gdzie jest /shared)
        project_root = os.path.dirname(os.path.abspath(__file__))
        while not os.path.exists(os.path.join(project_root, "shared")):
            project_root = os.path.dirname(project_root)
            if project_root == "/":
                raise FileNotFoundError("Nie znaleziono katalogu /shared")
        return project_root

    def prepare_features_from_files(
        self,
        rabbit_msg_id: str,
        train_file: str,
        test_file: str,
        fe_version: str = "fe_v2",
        feature_cols: list = None
    ) -> dict:
        """
        Przygotowuje pliki treningowe i testowe:
        - dodaje kolumny z FE
        - usuwa kolumny powiązane z targetem (log_return)
        - zapisuje pliki do katalogu processed_files
        - zachowuje kolumny z feature_cols + wszystkie kolumny FE
        """
        try:
            project_root = self._find_project_root()
            raw_dir = os.path.join(project_root, "shared", "files", "binance_historical_raw_data")
            processed_dir = self.PROCESSED_DIR

            # 🔹 Wczytaj pliki
            train_path_raw = os.path.join(raw_dir, train_file)
            test_path_raw = os.path.join(raw_dir, test_file)
            if not os.path.exists(train_path_raw):
                raise FileNotFoundError(f"Plik treningowy {train_file} nie istnieje w {raw_dir}")
            if not os.path.exists(test_path_raw):
                raise FileNotFoundError(f"Plik testowy {test_file} nie istnieje w {raw_dir}")

            df_train = pd.read_csv(train_path_raw)
            df_test = pd.read_csv(test_path_raw)

            # 🔹 FE (bez targetu)
            fe_module = importlib.import_module(f"sklearn_model.config.features.{fe_version}")
            FE = fe_module.FeatureEngineeringTransformer
            fe = FE()
            df_train = fe.transform(df_train)
            df_test = fe.transform(df_test)

            # 🔹 Usuń kolumny powiązane z targetem
            for df in [df_train, df_test]:
                if "log_return" in df.columns:
                    df.drop(columns=["log_return"], inplace=True)

            # 🔹 Zachowaj kolumny: feature_cols + wszystkie kolumny FE
            if feature_cols is not None:
                fe_columns = [col for col in df_train.columns if col not in feature_cols]
                cols_to_keep = feature_cols + fe_columns
                df_train = df_train[cols_to_keep]
                df_test = df_test[cols_to_keep]

            # 🔹 Zapis plików z prostymi nazwami train/test
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            train_filename = f"train_{date_str}.csv"
            test_filename = f"test_{date_str}.csv"

            train_path = os.path.join(processed_dir, train_filename)
            test_path = os.path.join(processed_dir, test_filename)

            df_train.to_csv(train_path, index=False)
            df_test.to_csv(test_path, index=False)

            print(f"✅ Pliki zapisane:\n - {train_path}\n - {test_path}")

            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "success",
                "message": "Train/test files prepared with features (log_return removed)",
                "train_filename": train_filename,
                "train_path": train_path,
                "test_filename": test_filename,
                "test_path": test_path,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "failed",
                "message": str(e),
                "train_filename": None,
                "train_path": None,
                "test_filename": None,
                "test_path": None,
                "timestamp": datetime.now().isoformat()
            }
