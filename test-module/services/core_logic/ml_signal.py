import joblib
import numpy as np
import os
from .standalone_model import ModelPredictor

class ModelSignal:
    def __init__(self, model_name, data, base_dir):
        # base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.BASE_MODEL_DIR = os.path.join(base_dir, "shared", "models")

        self.data = data
        self.model_name = model_name
        self.model_path = None
        print(f"\n[Core_Logic] Obliczanie predykcji.")
        match self.check_model_type():
            case "sklearn":
                self.model = joblib.load(self.model_path)
                self.predictions, self.classes = self.sklearn()

            case "lstm":
                self.model = ModelPredictor(self.model_path)
                self.predictions, self.classes = self.lstm_price_predictor()

    def sklearn(self):
        features = self.model.feature_names_in_
        x = self.data[features].values
        probs = self.model.predict_proba(x)
        classes = [str(c) for c in self.model.classes_]
        return probs, classes

    def lstm_price_predictor(self):
        preds = self.model.predict_batch(self.data)
        signals = []
        for p in preds:
            current = p["current_price"]
            predicted = p["predicted_price"]
            log_ret = np.log(predicted / current)
            conf = min(abs(log_ret) * 50, 1.0)
            if log_ret > 0:
                signals.append([conf, 1 - conf, 0.0])
            elif log_ret < 0:
                signals.append([0.0, 1 - conf, conf])
            else:
                signals.append([0.0, 1.0, 0.0])
        classes = ["buy", "hold", "sell"]
        return np.array(signals), classes




    def check_model_type(self):
        for subfolder in os.listdir(self.BASE_MODEL_DIR):
            subfolder_path = os.path.join(self.BASE_MODEL_DIR, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            candidate_path = os.path.join(subfolder_path, self.model_name)
            if os.path.exists(candidate_path):
                self.model_path = candidate_path
                return subfolder.lower()  # 'sklearn' lub 'lstm'

        raise ValueError(f"Nie znaleziono modelu '{self.model_name}' w żadnym z podfolderów {os.listdir(self.BASE_MODEL_DIR)}")