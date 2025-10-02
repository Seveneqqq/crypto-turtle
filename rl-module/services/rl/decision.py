from dataclasses import dataclass

@dataclass
class DecisionConfig:
    threshold_buy: float = 0.6
    threshold_sell: float = 0.6
    tx_fraction: float = 0.1
    max_fraction_when_confident: float = 0.5
    mincost: float = 1.0
    clamp_model_value: bool = True

class DecisionEngine:
    def __init__(self, config: DecisionConfig = DecisionConfig()):
        self.config = config

    def _normalize_model_value(self, mv):
        mv = float(mv)
        if self.config.clamp_model_value:
            mv = max(0.0, min(1.0, mv))
        return mv

    def decide(self, prob_map, wallet, symbol, price):
        buy_p = self._normalize_model_value(prob_map.get("buy", 0.0))
        sell_p = self._normalize_model_value(prob_map.get("sell", 0.0))

        if buy_p >= self.config.threshold_buy:
            action = "buy"
            model_value = buy_p
        elif sell_p >= self.config.threshold_sell:
            action = "sell"
            model_value = sell_p
        else:
            return {"action": "hold", "qty": 0.0, "model_value": 0.0}

        if action == "buy":
            available_cash = wallet.balance
            denom = 1.0 - self.config.threshold_buy
            frac = self.config.tx_fraction + ((model_value - self.config.threshold_buy) / denom) * (self.config.max_fraction_when_confident - self.config.tx_fraction)
            frac = max(0.0, min(1.0, frac))
            usd_amount = max(available_cash * frac, self.config.mincost)
            qty = usd_amount / price if price > 0 else 0.0
            return {"action": "buy", "qty": qty, "model_value": model_value}

        elif action == "sell":
            holding = wallet.holdings.get(symbol, 0.0)
            if holding <= 0:
                return {"action": "hold", "qty": 0.0, "model_value": model_value}
            denom = 1.0 - self.config.threshold_sell
            frac = self.config.tx_fraction + ((model_value - self.config.threshold_sell) / denom) * (self.config.max_fraction_when_confident - self.config.tx_fraction)
            frac = max(0.0, min(1.0, frac))
            qty = holding * frac
            return {"action": "sell", "qty": qty, "model_value": model_value}
