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
    def __init__(self, agent_name, rlagent_deterministic, wallet, config: DecisionConfig = DecisionConfig()):

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.BASE_MODEL_DIR = os.path.join(base_dir, "shared", "agents")
        self.agent_name = agent_name
        self.agent_path = ""
        self.config = config

        self.wallet = wallet
        self.rlagent_deterministic = rlagent_deterministic
        self.agent = None

        if self.agent_name:
            agent_type = self.check_agent_type()
            match agent_type:
                case "ppo":
                    self.agent = PPO.load(self.agent_path, env=None)
                case _:
                    raise ValueError(f"Unsupported agent type: {agent_type}")


    def _normalize_model_value(self, mv):
        if mv is None:
            return 0.0
        mv = float(mv)
        if self.config.clamp_model_value:
            mv = max(0.0, min(1.0, mv))
        return mv
    def check_agent_type(self):
        for subfolder in os.listdir(self.BASE_MODEL_DIR):
            subfolder_path = os.path.join(self.BASE_MODEL_DIR, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            candidate_path = os.path.join(subfolder_path, self.agent_name)
            if os.path.exists(candidate_path):
                self.agent_path = candidate_path
                return subfolder.lower()  # 'ppo' or other agent types

        raise ValueError(f"Nie znaleziono agenta '{self.agent_name}' w żadnym z podfolderów {os.listdir(self.BASE_MODEL_DIR)}")

    def rl_config(self, symbol, prob_vec, data_row):


        obs = self.build_obs(symbol, data_row, prob_vec)

        action, _ = self.agent.predict(obs, deterministic=self.rlagent_deterministic)
        action = np.asarray(action, dtype=np.float32).flatten()

        self.config.threshold_buy = float(action[0])
        self.config.threshold_sell = float(action[1])
        self.config.tx_fraction = float(action[2])
        self.config.max_fraction_when_confident = float(action[3])
        self.wallet.mincost = float(action[4])
        self.wallet.max_wallet_percentage_crypto = float(action[5])
        #self.wallet.stop_loss = float(action[6])

    def generate_decisions(self, prob_vec, symbols, prices, t, data_rows=None):
        use_rl = data_rows is not None and self.agent_name is not None
        
        orders = []

        for symbol in symbols:
            prob_map = prob_vec[symbol]
            best_action = max(prob_map, key=lambda k: prob_map[k])
            best_value = round(prob_map[best_action], 4)
            best_value = self._normalize_model_value(best_value)

            if use_rl:
                self.rl_config(symbol, prob_vec, data_rows[symbol])
                reason = "rl_adjusted"
            else:
                reason = "max_for_symbol"
            
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
        return round(frac,4)

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
