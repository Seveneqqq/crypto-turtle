import os
import joblib
import importlib
from datetime import datetime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class SklearnTrainService:
    def __init__(self, shared_dir: str = None):
        # katalog /shared
        if shared_dir is None:
            shared_dir = self._find_project_root()
        self.SHARED_DIR = os.path.join(shared_dir, "shared", "models", "sklearn")
        os.makedirs(self.SHARED_DIR, exist_ok=True)
        print(f"[SklearnTrainService] Modele będą zapisywane w: {self.SHARED_DIR}")

    @staticmethod
    def _find_project_root():
        # szuka katalogu głównego (tam gdzie jest /shared)
        project_root = os.path.dirname(os.path.abspath(__file__))
        while not os.path.exists(os.path.join(project_root, "shared")):
            project_root = os.path.dirname(project_root)
            if project_root == "/":
                raise FileNotFoundError("Nie znaleziono katalogu /shared")
        return project_root

    def get_model_path(self, base_name: str):
        existing_files = [f for f in os.listdir(self.SHARED_DIR) if f.startswith(base_name)]
        if existing_files:
            indices = [int(f.split("_")[-1].split(".")[0]) for f in existing_files]
            idx = max(indices) + 1
        else:
            idx = 0
        model_name = f"{base_name}_{idx}.pkl"
        return os.path.join(self.SHARED_DIR, model_name)

    def train_model(self, 
                    rabbit_msg_id,
                    df_filename: str,
                    fe_version: str = "fe_v3",
                    target_version: str = "t_v1",
                    clf_version: str = "c_v1",
                    feature_cols: list = None,
                    target_col: str = "target_cls") -> str:
        """
        Trenuje model i zapisuje pipeline w shared/models/sklearn
        Zwraca pełną ścieżkę do zapisanego modelu.
        """
        try:

            project_root = self._find_project_root()
            processed_dir = os.path.join(project_root, "shared", "files", "processed_files")
            df_path = os.path.join(processed_dir, df_filename)

            if not os.path.exists(df_path):
                raise FileNotFoundError(f"File {df_filename} doesn't exists in {processed_dir}")

            feature_cols = feature_cols or ["open", "high", "low", "close", "volume"]

            # 🔹 2. Wczytanie danych
            df = pd.read_csv(df_path)

            # 2️⃣ Dynamiczny import konfiguracji
            fe_module = importlib.import_module(f"sklearn_model.config.features.{fe_version}")
            FE = fe_module.FeatureEngineeringTransformer

            target_module = importlib.import_module(f"sklearn_model.config.target.{target_version}")
            df = target_module.create_target(df)

            clf_module = importlib.import_module(f"sklearn_model.config.classifier.{clf_version}")
            clf = clf_module.get_classifier()

            # 3️⃣ Przygotowanie danych
            df.dropna(subset=feature_cols + [target_col], inplace=True)
            df = df.iloc[50:]  # opcjonalnie, jeśli chcesz usunąć początkowe wiersze

            X_train = df[feature_cols]
            y_train = df[target_col]

            # 4️⃣ Pipeline
            pipeline = Pipeline([
                ("features", FE()),
                ("imputer", SimpleImputer(strategy="mean")),
                ("model", clf)
            ])

            # 5️⃣ Trenowanie
            pipeline.fit(X_train, y_train)

            # 6️⃣ Generowanie nazwy modelu
            date_str = datetime.now().strftime("%d%m")
            base_name = f"btc_model_{fe_version}_{target_version}_{clf_version}_{date_str}"
            model_path = self.get_model_path(base_name)

            # 7️⃣ Zapis pipeline
            joblib.dump(pipeline, model_path)
            print(f"✅ Pipeline zapisany jako {model_path}")

            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "success",
                "message": "Successfully trained model",
                "filename": base_name,
                "path": model_path,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "failed",
                "message": str(e),
                "filename": None,
                "path": None,
                "timestamp": datetime.now().isoformat()
            }