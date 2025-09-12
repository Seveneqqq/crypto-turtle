import os
import joblib
import pandas as pd
import numpy as np
import importlib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

class SklearnTestService:
    def __init__(self, shared_dir: str = None):
        # katalog /shared
        if shared_dir is None:
            shared_dir = self._find_project_root()
        self.SHARED_DIR = os.path.join(shared_dir, "models", "sklearn")
        self.TEST_DATA_DIR = os.path.join(shared_dir, "files", "btc_test_data")
        print(f"[SklearnTestService] Modele w: {self.SHARED_DIR}")
        print(f"[SklearnTestService] Dane testowe w: {self.TEST_DATA_DIR}")

    @staticmethod
    def _find_project_root():
        project_root = os.path.dirname(os.path.abspath(__file__))
        while not os.path.exists(os.path.join(project_root, "shared")):
            project_root = os.path.dirname(project_root)
            if project_root == "/":
                raise FileNotFoundError("Nie znaleziono katalogu /shared")
        return project_root

    def get_model_path(self, model_name: str) -> str:
        return os.path.join(self.SHARED_DIR, model_name)

    def get_test_data_path(self, filename: str) -> str:
        return os.path.join(self.TEST_DATA_DIR, filename)

    def test_model(self, model_name: str, data_file: str, cv_splits: int = 3, target_version: str = "t_v1"):
        """
        Testuje model z pliku model_name na danych testowych data_file
        i wykonuje CV z TimeSeriesSplit.
        Zwraca słownik z accuracy, raportem klasyfikacji i macierzą pomyłek.
        """
        model_path = self.get_model_path(model_name)
        data_path = self.get_test_data_path(data_file)

        # Wczytanie modelu
        pipeline = joblib.load(model_path)

        # Wczytanie danych testowych
        df = pd.read_csv(data_path)

        # Target
        target_module = importlib.import_module(f"config.target.{target_version}")
        df = target_module.create_target(df)

        feature_cols = ["open", "high", "low", "close", "volume"]
        target_col = "target_cls"

        df.dropna(subset=feature_cols + [target_col], inplace=True)
        X_test = df[feature_cols]
        y_test = df[target_col]

        # CV
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        y_pred_all = []
        y_true_all = []

        for train_idx, test_idx in tscv.split(X_test):
            X_tr, X_te = X_test.iloc[train_idx], X_test.iloc[test_idx]
            y_tr, y_te = y_test.iloc[train_idx], y_test.iloc[test_idx]

            # Fit pipeline na foldzie
            pipeline.fit(X_tr, y_tr)
            y_pred_fold = pipeline.predict(X_te)

            y_pred_all.extend(y_pred_fold)
            y_true_all.extend(y_te)

        y_pred_all = np.array(y_pred_all)
        y_true_all = np.array(y_true_all)

        # Accuracy
        acc = accuracy_score(y_true_all, y_pred_all)
        class_labels = ["sell", "hold", "buy"]
        report = classification_report(y_true_all, y_pred_all,
                                       labels=class_labels,
                                       target_names=class_labels,
                                       zero_division=0)
        cm = confusion_matrix(y_true_all, y_pred_all, labels=class_labels)

        results = {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": pd.DataFrame(cm,
                                             index=[f"true_{c}" for c in class_labels],
                                             columns=[f"pred_{c}" for c in class_labels]).to_dict(),
            "n_splits": cv_splits
        }

        print(f"\n✅ Accuracy ({cv_splits}-fold TimeSeriesSplit): {acc:.4f}")
        print(f"\n📊 Raport klasyfikacji ({cv_splits}-fold):\n{report}")
        print(f"\n📌 Macierz pomyłek ({cv_splits}-fold):\n{results['confusion_matrix']}")

        return results
