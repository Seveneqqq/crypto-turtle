from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os, joblib
import pandas as pd
from datetime import datetime
import importlib


class SklearnTrainServiceSimple:
    def __init__(self, shared_dir: str = None):
        if shared_dir is None:
            shared_dir = self._find_project_root()
        self.SHARED_DIR = os.path.join(shared_dir, "shared", "models", "sklearn")
        os.makedirs(self.SHARED_DIR, exist_ok=True)
        print(f"[SklearnTrainServiceSimple] Modele będą zapisywane w: {self.SHARED_DIR}")

    @staticmethod
    def _find_project_root():
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
                    feature_cols: list = None,
                    clf_version: str = "c_v1") -> dict:

        try:
            project_root = self._find_project_root()
            processed_dir = os.path.join(project_root, "shared", "files", "processed_trainmodel_data")
            df_path = os.path.join(processed_dir, df_filename)

            if not os.path.exists(df_path):
                raise FileNotFoundError(f"File {df_filename} doesn't exist in {processed_dir}")

            df = pd.read_csv(df_path)

            # 🔹 Tworzenie targetu w serwisie
            target_module = importlib.import_module("sklearn_model.config.target.t_v1")
            df = target_module.create_target(df)
            target_col = "target_cls"

            feature_cols = feature_cols or [c for c in df.columns if c not in [target_col, "future_log_return"]]

            df.dropna(subset=feature_cols + [target_col], inplace=True)

            X_train = df[feature_cols]
            y_train = df[target_col]

            # 🔹 Pipeline: imputer + scaler + model
            clf_module = importlib.import_module(f"sklearn_model.config.classifier.{clf_version}")
            clf = clf_module.get_classifier()

            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                # ("scaler", StandardScaler()),
                ("model", clf)
            ])

            pipeline.fit(X_train, y_train)

            date_str = datetime.now().strftime("%d%m")
            base_name = f"btc_model_simple_{clf_version}_{date_str}"
            model_path = self.get_model_path(base_name)

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
