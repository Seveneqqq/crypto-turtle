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
    def __init__(self, agent_name,wallet, config: DecisionConfig = DecisionConfig()):

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.BASE_MODEL_DIR = os.path.join(base_dir, "shared", "agents")
        self.agent_name = agent_name
        self.agent_path = None
        self.config = config

        self.wallet = wallet
        # self.check_model_type()
        # match self.check_model_type():
        #     case "ppo":
        #         self.agent = PPO.load(self.agent_path, env=None)
        #
        #     case "dqn":
        #         pass


    def _normalize_model_value(self, mv):
        if mv is None:
            return 0.0
        mv = float(mv)
        if self.config.clamp_model_value:
            mv = max(0.0, min(1.0, mv))
        return mv
    def rl_generate_decisions(self):
        pass

    def ml_generate_decisions(self, prob_vec, symbols, prices, t):
        orders = []

        for symbol in symbols:
            prob_map = prob_vec[symbol]
            best_action = max(prob_map, key=lambda k: prob_map[k])
            best_value = round(prob_map[best_action],4)
            best_value = self._normalize_model_value(best_value)
            orders.append({
                "symbol": symbol,
                "action": best_action,
                "value": best_value,
                "price": prices[symbol],
                "timestamp": t,
                "reason": "max_for_symbol"
            })

        for order in orders:
            action = order["action"]
            value = order["value"]

            match action:
                case "buy":
                    if value < self.config.threshold_buy:
                        order["action"] = "hold"
                        order["value"]  = 1.0
                        order["reason"] = "below_threshold"
                    else:
                        order["fraction"] = self.calculate_fraction(value, self.config.threshold_buy)

                case "sell":
                    if value < self.config.threshold_sell:
                        order["action"] = "hold"
                        order["value"]  = 1.0
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
        return round(frac,4)

    def build_obs(self, symbol, t, data, rl_env_wallet, prob_array, classes):
        price = float(data[symbol].iloc[t]["close"])
        prob_vec = prob_array  # numpy array
        row = data[symbol].iloc[t]
        # Mapowanie nazw klas na indeksy w tablicy
        class_idx = {c: i for i, c in enumerate(classes)}
        price_norm = price / (107312.78 + 1e-12)
        balance_norm = (float(rl_env_wallet.balance) / (1000.0 + 1e-12))
        buy_p = float(prob_vec[class_idx["buy"]])
        sell_p = float(prob_vec[class_idx["sell"]])
        hold_p = float(prob_vec[class_idx["hold"]])
        balance = float(rl_env_wallet.balance)
        ret1 = float(row.get("ret1", 0.0))
        ret1_lag1 = float(row.get("ret1_lag1", 0.0))
        roll_mean_10 = float(row.get("roll_mean_10", 0.0))
        roll_std_10 = float(row.get("roll_std_10", 0.0))
        macd = float(row.get("macd", 0.0))
        macd_sig = float(row.get("macd_sig", 0.0))
        atr = float(row.get("ATR_14", 0.0))
        vol = float(row.get("volume", 0.0))
        obs = np.array([
            buy_p, sell_p, price_norm, balance_norm,
            ret1, ret1_lag1, roll_mean_10, roll_std_10, macd, macd_sig, atr, vol
        ], dtype=np.float32)
        return obs

    def check_model_type(self):
        for subfolder in os.listdir(self.BASE_MODEL_DIR):
            subfolder_path = os.path.join(self.BASE_MODEL_DIR, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            candidate_path = os.path.join(subfolder_path, self.agent_name)
            if os.path.exists(candidate_path):
                self.agent_path = candidate_path
                return subfolder.lower()  # 'sklearn' lub 'lstm'

        raise ValueError(f"Nie znaleziono modelu '{self.agent_name}' w żadnym z podfolderów {os.listdir(self.BASE_MODEL_DIR)}")
