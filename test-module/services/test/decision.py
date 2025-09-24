import joblib
import pandas as pd


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


    def decide(self, prob_row):

        try:
            # Sprawdzenie typu
            if isinstance(prob_row, pd.Series):
                prob_map = prob_row.to_dict()
            elif isinstance(prob_row, dict):
                prob_map = prob_row
            else:
                raise ValueError("Nieobsługiwany typ danych wejściowych. Oczekiwano dict lub pd.Series")

            # Decyzja na podstawie progów
            if prob_map.get("buy", 0) >= self.threshold_buy:
                return prob_map.get("buy", 0), "buy"
            elif prob_map.get("sell", 0) >= self.threshold_sell:
                return prob_map.get("sell", 0), "sell"
            else:
                return 0.0, "hold"

        except Exception as e:
            import traceback
            print("[DecisionEngine.decide][ERROR]", e)
            traceback.print_exc()
            raise