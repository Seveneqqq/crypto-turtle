import os
from dataclasses import dataclass

@dataclass
class DecisionConfig:
    threshold_buy: float = 0.6
    threshold_sell: float = 0.6
    tx_fraction: float = 0.1
    max_fraction_when_confident: float = 0.5
    clamp_model_value: bool = True

class DecisionEngine:
    def __init__(self, wallet, config: DecisionConfig = DecisionConfig()):

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.BASE_MODEL_DIR = os.path.join(base_dir, "shared", "agents")
        self.config = config
        self.wallet = wallet

    def _normalize_model_value(self, mv):
        if mv is None:
            return 0.0
        mv = float(mv)
        if self.config.clamp_model_value:
            mv = max(0.0, min(1.0, mv))
        return mv

    def rl_decision_config(self, scaled):
        thr_low, thr_high = 0.2, 0.8
        threshold_buy = thr_low + scaled[0] * (thr_high - thr_low)
        threshold_sell = thr_low + scaled[1] * (thr_high - thr_low)

        tx_min, tx_max = 0.001, 0.2
        tx_fraction = tx_min + scaled[2] * (tx_max - tx_min)

        tx_min2, tx_max2 = 0.001, 0.2
        tx_max_fraction = tx_min2 + scaled[3] * (tx_max2 - tx_min2)

        self.config.threshold_buy = float(threshold_buy)
        self.config.threshold_sell = float(threshold_sell)
        self.config.tx_fraction = float(tx_fraction)
        self.config.max_fraction_when_confident = float(tx_max_fraction)

    def generate_decisions(self, prob_vec, symbols, prices, t, scaled_dict):
        orders = []

        for symbol in symbols:
            prob_map = prob_vec[symbol]
            best_action = max(prob_map, key=lambda k: prob_map[k])
            best_value = round(prob_map[best_action], 4)
            best_value = self._normalize_model_value(best_value)

            orders.append({
                "symbol": symbol,
                "action": best_action,
                "value": best_value,
                "price": prices[symbol],
                "timestamp": t,
                "reason": "rl_adjusted"
            })

        for order in orders:
            action = order["action"]
            value = order["value"]
            symbol = order["symbol"]

            order["rl_config"] = scaled_dict[symbol]
            self.rl_decision_config(scaled_dict[symbol])

            match action:
                case "buy":
                    if value < self.config.threshold_buy:
                        order["action"] = "hold"
                        order["value"] = 1.0
                        order["reason"] = "below_threshold"
                    else:
                        order["fraction"] = self.calculate_fraction(value, self.config.threshold_buy)

                case "sell":
                    if value < self.config.threshold_sell:
                        order["action"] = "hold"
                        order["value"] = 1.0
                        order["reason"] = "below_threshold"
                    else:
                        order["fraction"] = self.calculate_fraction(value, self.config.threshold_sell)
                case "hold":
                    pass

        orders.sort(key=lambda o: o["value"], reverse=True)

        self.wallet.orders.extend(orders)
        self.wallet.execute_orders(t, prices)

    def calculate_fraction(self, model_value, threshold):
        denom = max(1.0 - threshold, 1e-12)

        norm_signal = max(0.0, min(1.0, (model_value - threshold) / denom))

        frac = self.config.tx_fraction + norm_signal * (
                self.config.max_fraction_when_confident - self.config.tx_fraction)

        frac = max(0.0, min(1.0, frac))
        return round(frac, 4)