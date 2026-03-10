import os
import pandas as pd
import numpy as np
from datetime import datetime
import importlib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import randint


class SklearnHyperparamTuningService:
    def __init__(self, shared_dir: str = None):
        if shared_dir is None:
            shared_dir = self._find_project_root()
        self.LOG_DIR = os.path.join(shared_dir, "shared", "models", "tuning_logs")
        os.makedirs(self.LOG_DIR, exist_ok=True)
        print(f"[SklearnHyperparamTuningService] Logi będą zapisywane w: {self.LOG_DIR}")

    @staticmethod
    def _find_project_root():
        project_root = os.path.dirname(os.path.abspath(__file__))
        while not os.path.exists(os.path.join(project_root, "shared")):
            project_root = os.path.dirname(project_root)
            if project_root == "/":
                raise FileNotFoundError("Nie znaleziono katalogu /shared")
        return project_root

    def _load_data(self, filename: str):
        """Ładuje dane i tworzy kolumnę target."""
        project_root = self._find_project_root()
        processed_dir = os.path.join(project_root, "shared", "files", "processed_trainmodel_data")
        df_path = os.path.join(processed_dir, filename)

        if not os.path.exists(df_path):
            raise FileNotFoundError(f"File {filename} doesn't exist in {processed_dir}")

        df = pd.read_csv(df_path)

        # Tworzymy kolumnę target
        target_module = importlib.import_module("sklearn_model.config.target.t_v1")
        df = target_module.create_target(df)
        df.dropna(inplace=True)

        target_col = "target_cls"
        feature_cols = [c for c in df.columns if c not in [target_col, "future_log_return"]]
        return df[feature_cols], df[target_col]

    def tune_random_forest(
        self,
        df_train_filename: str,
        df_test_filename: str,
        n_iter: int = 50,
        cv_folds: int = 3,
        random_seed: int = 42,
    ):
        """Tuning RandomForest z bieżącym zapisem postępu do logów."""
        np.random.seed(random_seed)

        X_train, y_train = self._load_data(df_train_filename)
        X_test, y_test = self._load_data(df_test_filename)

        # 🔹 Kolumny numeryczne
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train = X_train[num_cols]
        X_test = X_test[num_cols]

        # 🔹 RandomForest + Pipeline
        clf = RandomForestClassifier(random_state=random_seed, n_jobs=-1)
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean"))
        ])
        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, num_cols)],
            remainder="drop"
        )
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", clf)
        ])

        # 🔹 Zakres hiperparametrów
        param_dist = {
            "model__n_estimators": randint(100, 2000),
            "model__max_depth": [None, 5, 10, 20, 30, 40, 50],
            "model__min_samples_split": randint(2, 20),
            "model__min_samples_leaf": randint(1, 10),
            "model__max_features": ["sqrt", "log2", None, 0.3, 0.5, 0.8],
            "model__bootstrap": [True, False],
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_log_path = os.path.join(self.LOG_DIR, f"tuning_RF_{timestamp}.csv")
        progress_log_path = base_log_path.replace(".csv", "_progress.csv")

        print(f"🚀 Start RandomizedSearchCV ({n_iter} prób, {cv_folds}-fold CV, n_jobs=1)")

        # 🔹 RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv_folds,
            verbose=2,
            n_jobs=1,
            random_state=random_seed,
            scoring="f1_macro",
            return_train_score=True
        )

        search.fit(X_train, y_train)

        # 🔹 Zapis pełnych wyników
        results_df = pd.DataFrame(search.cv_results_)
        results_df.to_csv(base_log_path, index=False)
        print(f"✅ Pełne wyniki zapisano: {base_log_path}")

        # 🔹 Bieżący zapis wyników (progress)
        progress_cols = ["params", "mean_test_score", "rank_test_score"]
        progress_df = results_df[progress_cols].sort_values(by="rank_test_score")
        progress_df.to_csv(progress_log_path, index=False)
        print(f"📊 Postęp zapisano: {progress_log_path}")

        # 🔹 Ewaluacja najlepszego modelu
        best_model = search.best_estimator_
        preds = best_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        top10_path = base_log_path.replace(".csv", "_TOP10.csv")
        results_df.sort_values(by="mean_test_score", ascending=False).head(10).to_csv(top10_path, index=False)

        print("🎯 Wyniki tuningu:")
        print(f"Najlepsze parametry: {search.best_params_}")
        print(f"Średni F1 z CV: {search.best_score_:.4f}")
        print(f"Dokładność (test): {acc:.4f}")
        print(f"F1 (test): {f1:.4f}")
        print(f"Top 10 zapisano: {top10_path}")

        return {
            "result": "success",
            "best_params": search.best_params_,
            "best_cv_f1": float(search.best_score_),
            "test_accuracy": float(acc),
            "test_f1": float(f1),
            "log_file": base_log_path,
            "progress_file": progress_log_path,
            "top10_file": top10_path,
        }
