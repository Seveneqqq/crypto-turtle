import joblib
import numpy as np
import pandas as pd
import os

class DecisionEngine:
    def __init__(self, model_path, threshold_buy=0.6, threshold_sell=0.6):
        # Ładowanie modelu przy starcie
        try:
            print(f"[DecisionEngine.__init__] Ładuję model: {model_path}")
            self.model = joblib.load(model_path)
            print(f"[DecisionEngine.__init__] Model załadowany. Classes={getattr(self.model, 'classes_', None)}")
        except Exception as e:
            import traceback
            print("[DecisionEngine.__init__][ERROR]", e)
            traceback.print_exc()
            raise
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell

    def decide(self, features):
        # Konwersja do numpy array
        try:
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
            if prob_map.get("buy", 0) >= self.threshold_buy:
                a = prob_map.get("buy", 0)
                return a, "buy"
            elif prob_map.get("sell", 0) >= self.threshold_sell:
                b = prob_map.get("sell", 0)
                return b, "sell"
            else:
                return 0.0, "hold"


        except Exception as e:
            import traceback
            print("[DecisionEngine.decide][ERROR]", e)
            print("[DecisionEngine.decide] features type=", type(features))
            if isinstance(features, (pd.Series, pd.DataFrame)):
                try:
                    print("[DecisionEngine.decide] features columns=", getattr(features, 'columns', None))
                except Exception:
                    pass
            traceback.print_exc()
            raise