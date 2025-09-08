import joblib
import numpy as np
import pandas as pd
import os

class DecisionEngine:
    def __init__(self, model_path, threshold=0.6):
        # Ładowanie modelu przy starcie
        self.model = joblib.load(model_path)
        self.threshold = threshold

    def decide(self, features):
        # Konwersja do numpy array
        if isinstance(features, dict):
            x = np.array([list(features.values())])
        elif isinstance(features, np.ndarray):
            x = features.reshape(1, -1)
        elif isinstance(features, pd.Series):
            x = features.values.reshape(1, -1)
        elif isinstance(features, pd.DataFrame):
            x = features  # pojedynczy wiersz
        else:
            raise ValueError("Nieobsługiwany typ danych wejściowych dla DecisionEngine")

        # Prawdopodobieństwa klas
        probs = self.model.predict_proba(x)[0]
        prob_map = dict(zip(self.model.classes_, probs))

        # Decyzja na podstawie progów
        if prob_map.get("buy", 0) >= self.threshold:
            a = prob_map.get("buy", 0)
            return a, "buy"
        elif prob_map.get("sell", 0) >= self.threshold:
            b = prob_map.get("sell", 0)
            return b, "sell"
        else:
            return 0.0, "hold"