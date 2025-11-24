# decision.py
import os
import numpy as np
from dataclasses import dataclass
from stable_baselines3 import PPO


@dataclass
class DecisionConfig:
    threshold_buy: float = 0.6
    threshold_sell: float = 0.6
    tx_fraction: float = 0.1  # fraction of cash to use for buy when signal strong
    max_fraction_when_confident: float = 0.5  # if model gives 0.99 use up to 50% cash
    clamp_model_value: bool = True


class DecisionEngine:
    def __init__(self, agent_name, wallet, rlagent_deterministic=False, config: DecisionConfig = DecisionConfig()):

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.BASE_MODEL_DIR = os.path.join(base_dir, "shared", "agents")
        self.agent_name = agent_name
        self.agent_path = ""
        self.config = config

        self.wallet = wallet
        self.rlagent_deterministic = rlagent_deterministic
        self.agent = None

    def _normalize_model_value(self, mv):
        if mv is None:
            return 0.0
        mv = float(mv)
        if self.config.clamp_model_value:
            mv = max(0.0, min(1.0, mv))
        return mv

    def rl_decision_config(self, scaled):
        # mapping
        # thresholds w zakresie [0.2, 0.8]
        thr_low, thr_high = 0.2, 0.8
        threshold_buy = thr_low + scaled[0] * (thr_high - thr_low)
        threshold_sell = thr_low + scaled[1] * (thr_high - thr_low)

        # tx_fraction w zakresie [0.001, 0.2] (0.1% - 20% kapitału)
        tx_min, tx_max = 0.001, 0.2
        tx_fraction = tx_min + scaled[2] * (tx_max - tx_min)

        # max_fraction w zakresie [0.1, 0.6]
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

            reason = "rl_adjusted"

            orders.append({
                "symbol": symbol,
                "action": best_action,
                "value": best_value,
                "price": prices[symbol],
                "timestamp": t,
                "reason": reason
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
        # frac mincost, qty
        return round(frac, 4)

    def build_obs(self, symbol, data_rows, prob_vec):

        numeric_cols = data_rows.select_dtypes(include=[np.number]).columns
        data_values = data_rows[numeric_cols].values.astype(np.float32)

        buy_p = float(prob_vec[symbol]["buy"])
        hold_p = float(prob_vec[symbol]["hold"])
        sell_p = float(prob_vec[symbol]["sell"])
        balance = float(self.wallet.balance)
        holdings = float(self.wallet.holdings.get(symbol, 0.0))

        obs = np.concatenate([
            np.array([buy_p, sell_p, hold_p], dtype=np.float32),
            data_values,
            np.array([balance, holdings], dtype=np.float32)
        ])

        return obs
