import joblib
import numpy as np
import os
from services.test.standalone_model import ModelPredictor

class ModelSignal:
    def __init__(self, model_path, data):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.MODEL_DIR = os.path.join(base_dir, "shared", "models", "sklearn")
        self.data = data
        self.model_path = model_path

        match self.check_model_type():
            case "sklearn":
                self.model = joblib.load(model_path)
                self.predictions, self.classes = self.sklearn()

            case "lstm":
                self.model = ModelPredictor(model_path)
                self.predictions, self.classes = self.lstm_price_predictor()

    def sklearn(self):
        features = self.model.feature_names_in_
        x = self.data[features].values
        probs = self.model.predict_proba(x)
        classes = [str(c) for c in self.model.classes_]
        return probs, classes

    def lstm_price_predictor(self):
        predictor = ModelPredictor(self.model)
        preds = predictor.predict_batch(self.data)
        # Przekształcamy w „probabilities” zgodne z decision engine
        signals = []
        for p in preds:
            current = p["current_price"]
            predicted = p["predicted_price"]
            log_ret = np.log(predicted / current)
            conf = min(abs(log_ret) * 50, 1.0)  # np. 2% zmiany ≈ 0.02*50 = 1.0
            if log_ret > 0:
                signals.append([conf, 1 - conf, 0.0])
            elif log_ret < 0:
                signals.append([0.0, 1 - conf, conf])
            else:
                signals.append([0.0, 1.0, 0.0])
        classes = ["buy", "hold", "sell"]
        return np.array(signals), classes




    def check_model_type(self):
        full_path = os.path.join(self.MODEL_DIR, self.model_path)

        if not os.path.exists(full_path):
            raise ValueError(f"Ścieżka nie istnieje: {full_path}")

        if os.path.isdir(full_path):
            return "lstm"

        filename = os.path.basename(full_path)
        if "." in filename:
            return "sklearn"

        raise ValueError(
            f"Nie udało się określić typu modelu (brak rozszerzenia i nie jest to folder): {self.model_path}"
        )
