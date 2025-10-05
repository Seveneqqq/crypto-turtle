import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from datetime import datetime

class AccuracyService:

    def __init__(self):
        self.base_dir = os.getcwd()

    def calculate_accuracy(self, rabbit_msg_id=None, data_dir=None, data_file=None, model_dir=None, model_file=None):
        try:
            # === 1. Ścieżki ===
            data_path = os.path.join(self.base_dir, "shared", "files", data_dir, data_file)
            model_path = os.path.join(self.base_dir, "shared", "models", model_dir, model_file)

            # === 2. Wczytaj dane (CSV) ===
            df = pd.read_csv(data_path)

            # === 2a. Obsługa kolumn daty i stringów ===
            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        # Spróbuj przekonwertować na datetime
                        df[col] = pd.to_datetime(df[col])
                        df[col + "_year"] = df[col].dt.year
                        df[col + "_month"] = df[col].dt.month
                        df[col + "_day"] = df[col].dt.day
                        df[col + "_hour"] = df[col].dt.hour
                        df[col + "_minute"] = df[col].dt.minute
                        df[col + "_second"] = df[col].dt.second
                        df.drop(col, axis=1, inplace=True)
                    except Exception:
                        # Jeśli nie da się przekonwertować na datetime, zamień na kody numeryczne
                        df[col] = df[col].astype('category').cat.codes

            # === 2b. Przygotowanie danych do modelu ===
            X_test = df.drop("target", axis=1).values
            y_true = df["target"].values

            # === 3. Wczytaj model ===
            model = joblib.load(model_path)

            # === 4. Predykcje i accuracy ===
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_true, y_pred)

            # === 5. Przygotowanie folderu accuracy ===
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            acc_dir = os.path.join(self.base_dir, "shared", "accuracy", timestamp)
            os.makedirs(acc_dir, exist_ok=True)

            # === 6. Zapis wyników ===
            result_data = {
                "result": "success",
                "accuracy": acc,
                "model_file": model_file,
                "data_file": data_file,
                "timestamp": timestamp,
                "rabbit_msg_id": rabbit_msg_id
            }

            # result.json
            result_path = os.path.join(acc_dir, "result.json")
            with open(result_path, "w") as f:
                json.dump(result_data, f, indent=4)

            # log.txt
            log_path = os.path.join(acc_dir, "log.txt")
            with open(log_path, "w") as f:
                f.write(f"[{timestamp}] Accuracy calculation log\n")
                f.write(f"Data file: {data_file}\n")
                f.write(f"Model file: {model_file}\n")
                f.write(f"Accuracy: {acc}\n")
                f.write("Status: success\n")

            return result_data

        except Exception as e:
            print(f"Error in calculate_accuracy: {e}")
            return {
                "result": "failed",
                "message": str(e),
                "rabbit_msg_id": rabbit_msg_id,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
