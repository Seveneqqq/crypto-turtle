import os
import joblib
import pandas as pd
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import importlib


class SklearnTestServiceSimple:
    def __init__(self, shared_dir: str = None):
        if shared_dir is None:
            shared_dir = self._find_project_root()
        self.SHARED_DIR = os.path.join(shared_dir, "shared")
        self.TEST_LOGS_DIR = os.path.join(self.SHARED_DIR, "models", "test_logs")
        os.makedirs(self.TEST_LOGS_DIR, exist_ok=True)
        print(f"[SklearnTestServiceSimple] Wspólny katalog: {self.SHARED_DIR}")
        print(f"[SklearnTestServiceSimple] Logi będą zapisywane w: {self.TEST_LOGS_DIR}")

    @staticmethod
    def _find_project_root():
        project_root = os.path.dirname(os.path.abspath(__file__))
        while not os.path.exists(os.path.join(project_root, "shared")):
            project_root = os.path.dirname(project_root)
            if project_root == "/":
                raise FileNotFoundError("Nie znaleziono katalogu /shared")
        return project_root

    def test_model(self,
                   rabbit_msg_id: str,
                   model_filename: str,
                   test_filename: str):
        try:
            # 🔹 Ścieżki
            model_path = os.path.join(self.SHARED_DIR, "models", "sklearn", model_filename)
            test_path = os.path.join(self.SHARED_DIR, "files", "processed_testmodel_data", test_filename)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model {model_filename} nie istnieje w {model_path}")
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Plik testowy {test_filename} nie istnieje w {test_path}")

            # 🔹 Wczytanie pipeline
            model = joblib.load(model_path)

            # 🔹 Wczytanie danych testowych
            df = pd.read_csv(test_path)

            # 🔹 Tworzenie targetu w serwisie
            target_module = importlib.import_module("sklearn_model.config.target.t_v1")
            df = target_module.create_target(df)
            target_col = "target_cls"

            if target_col not in df.columns:
                raise ValueError(f"Brak kolumny targetu '{target_col}' po utworzeniu targetu")

            # 🔹 Użycie tylko cech, które były podczas fit
            feature_cols = list(model.feature_names_in_)
            df = df[feature_cols + [target_col]].dropna(subset=feature_cols + [target_col])
            X_test = df[feature_cols]
            y_test = df[target_col]

            # 🔹 Predykcja
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

            # 🔹 Metryki
            acc = accuracy_score(y_test, y_pred)
            class_labels = list(model.classes_)

            report = classification_report(
                y_test,
                y_pred,
                labels=class_labels,
                target_names=class_labels,
                zero_division=0,
                output_dict=True
            )

            cm = confusion_matrix(y_test, y_pred, labels=class_labels).tolist()

            # 🔹 Przykładowe predykcje
            sample_predictions = []
            for i in range(min(20, len(X_test))):
                entry = {
                    "index": int(i),
                    "true": str(y_test.iloc[i]),
                    "predicted": str(y_pred[i]),
                    "features": {col: float(X_test.iloc[i][col]) for col in feature_cols}
                }
                if y_proba is not None:
                    proba_dict = {label: float(y_proba[i][j]) for j, label in enumerate(class_labels)}
                    entry["probabilities"] = proba_dict
                    entry["predicted_proba_max"] = max(proba_dict, key=proba_dict.get)
                sample_predictions.append(entry)

            # 🔹 Wynik
            result = {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "success",
                "message": "Model tested successfully (simple mode)",
                "model": model_filename,
                "test_file": test_filename,
                "metrics": {
                    "accuracy": acc,
                    "classification_report": report,
                    "confusion_matrix": {"labels": class_labels, "matrix": cm}
                },
                "sample_predictions": sample_predictions,
                "timestamp": datetime.now().isoformat()
            }

            # 🔹 Log
            log_filename = f"test_{os.path.splitext(model_filename)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            log_path = os.path.join(self.TEST_LOGS_DIR, log_filename)
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"📂 Log zapisany: {log_path}")

            return result

        except Exception as e:
            result = {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "failed",
                "message": str(e),
                "model": model_filename,
                "test_file": test_filename,
                "metrics": None,
                "timestamp": datetime.now().isoformat()
            }
            log_filename = f"test_ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            log_path = os.path.join(self.TEST_LOGS_DIR, log_filename)
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"❌ Błąd zapisany w logu: {log_path}")
            return result
