# import joblib
# decision.py
import joblib
import numpy as np
from dataclasses import dataclass

@dataclass
class DecisionConfig:
    threshold_buy: float = 0.6
    threshold_sell: float = 0.6
    sizing_mode: str = "mincost_based"   # or "mincost_based"
    mincost: float = 1.0
    tx_fraction: float = 0.1  # fraction of cash to use for buy when signal strong
    max_fraction_when_confident: float = 0.5  # if model gives 0.99 use up to 50% cash
    clamp_model_value: bool = True

class DecisionEngine:
    def __init__(self, model_path=None, config: DecisionConfig = DecisionConfig()):
        self.config = config
        if model_path:
            self.model = joblib.load(model_path)
        else:
            self.model = None

    def _normalize_model_value(self, mv):
        if mv is None:
            return 0.0
        mv = float(mv)
        if self.config.clamp_model_value:
            mv = max(0.0, min(1.0, mv))
        return mv

    def decide(self, prob_map, wallet, symbol, price):
        """
        prob_map: dict with keys 'buy', 'sell', 'hold' probabilities
        wallet: Wallet instance (to compute sizes)
        symbol, price: for sizing decisions
        returns: dict(action, qty, model_value, reason)
        """
        buy_p = prob_map.get("buy", 0.0)
        sell_p = prob_map.get("sell", 0.0)
        buy_p = self._normalize_model_value(buy_p)
        sell_p = self._normalize_model_value(sell_p)

        # Basic thresholding
        if buy_p >= self.config.threshold_buy:
            action = "buy"
            model_value = buy_p
        elif sell_p >= self.config.threshold_sell:
            action = "sell"
            model_value = sell_p
        else:
            return {"action": "hold", "qty": 0.0, "model_value": 0.0, "reason": "below thresholds"}

        # sizing
        if action == "buy":
            # fraction of available cash, scaled by model_value between threshold and 1.0
            available_cash = wallet.balance
            # map model_value from [threshold_buy,1] -> [tx_fraction, max_fraction_when_confident]
            denom = (1.0 - self.config.threshold_buy) if (1.0 - self.config.threshold_buy) > 0 else 1.0
            frac = self.config.tx_fraction + ((model_value - self.config.threshold_buy) / denom) * (self.config.max_fraction_when_confident - self.config.tx_fraction)
            frac = max(0.0, min(1.0, frac))
            usd_amount = available_cash * frac
            usd_amount = max(usd_amount, self.config.mincost)  # enforce mincost
            qty = usd_amount / price if price > 0 else 0.0
            return {"action": "buy", "qty": qty, "model_value": model_value, "reason": f"frac={frac:.3f}"}

        elif action == "sell":
            # amount to sell proportional to model_value
            holding = wallet.holdings.get(symbol, 0.0)
            if holding <= 0:
                return {"action": "hold", "qty": 0.0, "model_value": model_value, "reason": "no holdings to sell"}
            denom = (1.0 - self.config.threshold_sell) if (1.0 - self.config.threshold_sell) > 0 else 1.0
            frac = self.config.tx_fraction + ((model_value - self.config.threshold_sell) / denom) * (self.config.max_fraction_when_confident - self.config.tx_fraction)
            frac = max(0.0, min(1.0, frac))
            qty = holding * frac
            return {"action": "sell", "qty": qty, "model_value": model_value, "reason": f"frac={frac:.3f}"}



# import pandas as pd
#
#
# class DecisionEngine:
#     def __init__(self, model_path, threshold_buy=0.6, threshold_sell=0.6):
#         # Ładowanie modelu przy starcie
#         try:
#             print(f"[DecisionEngine.__init__] Ładuję model: {model_path}")
#             self.model = joblib.load(model_path)
#             print(f"[DecisionEngine.__init__] Model załadowany. Classes={getattr(self.model, 'classes_', None)}")
#         except Exception as e:
#             import traceback
#             print("[DecisionEngine.__init__][ERROR]", e)
#             traceback.print_exc()
#             raise
#         self.threshold_buy = threshold_buy
#         self.threshold_sell = threshold_sell
#
#
#     def decide(self, prob_row):
#
#         try:
#             # Sprawdzenie typu
#             if isinstance(prob_row, pd.Series):
#                 prob_map = prob_row.to_dict()
#             elif isinstance(prob_row, dict):
#                 prob_map = prob_row
#             else:
#                 raise ValueError("Nieobsługiwany typ danych wejściowych. Oczekiwano dict lub pd.Series")
#
#             # Decyzja na podstawie progów
#             if prob_map.get("buy", 0) >= self.threshold_buy:
#                 return prob_map.get("buy", 0), "buy"
#             elif prob_map.get("sell", 0) >= self.threshold_sell:
#                 return prob_map.get("sell", 0), "sell"
#             else:
#                 return 0.0, "hold"
#
#         except Exception as e:
#             import traceback
#             print("[DecisionEngine.decide][ERROR]", e)
#             traceback.print_exc()
#             raise