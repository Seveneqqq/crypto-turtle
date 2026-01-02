import os
import numpy as np
from dataclasses import dataclass
from stable_baselines3 import PPO, SAC


@dataclass
class DecisionConfig:
    threshold_buy: float = 0.6
    threshold_sell: float = 0.6
    tx_fraction: float = 0.1
    max_fraction_when_confident: float = 0.5
    clamp_model_value: bool = True

class DecisionEngine:
    def __init__(self, agent_name, rlagent_deterministic, wallet, base_dir, config: DecisionConfig = DecisionConfig()):
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
                case "sac":
                    self.agent = SAC.load(self.agent_path, env=None)
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
                return subfolder.lower()

        raise ValueError(f"Nie znaleziono agenta '{self.agent_name}' w żadnym z podfolderów {os.listdir(self.BASE_MODEL_DIR)}")

    def rl_config(self, scaled):

        self.config.threshold_buy = float(scaled[0])
        self.config.threshold_sell = float(scaled[1])
        self.config.tx_fraction = float(scaled[2])
        self.config.max_fraction_when_confident = float(scaled[3])


    def get_obs(self, symbols,data_rows, prob_vec, prices):
        scaled_dict = {}
        obs = self.build_obs(symbols, data_rows, prob_vec, prices)
        action, _ = self.agent.predict(obs, deterministic=self.rlagent_deterministic)
        action = np.asarray(action, dtype=np.float32).flatten()
        scaled = (action + 1.0) / 2.0
        #print(scaled)
        n_params_per_symbol = 6
        for i, symbol in enumerate(symbols):
            start = i * n_params_per_symbol
            end = start + n_params_per_symbol
            scaled_dict[symbol] = scaled[start:end]
        return scaled_dict


    def generate_decisions(self, prob_vec, symbols, prices, t, data_rows=None):
        use_rl = data_rows is not None and self.agent_name is not None
        
        orders, scaled_dict= [],{}
        if use_rl:
            scaled_dict = self.get_obs(symbols, data_rows, prob_vec, prices)


        for symbol in symbols:
            prob_map = prob_vec[symbol]
            best_action = max(prob_map, key=lambda k: prob_map[k])
            best_value = round(prob_map[best_action], 4)
            best_value = self._normalize_model_value(best_value)
            
            orders.append({
                "symbol": symbol,
                "predicted": best_action,
                "action": best_action,
                "value": best_value,
                "price": prices[symbol],
                "timestamp": t,
                "reason": "max_for_symbol"
            })

        for order in orders:
            action = order["action"]
            value = order["value"]
            symbol = order["symbol"]

            if use_rl:
                order["rl_config"] = scaled_dict[symbol]
                self.rl_config(scaled_dict[symbol])

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
        self.wallet.execute_orders(t, prices, use_rl)

    def calculate_fraction(self, model_value, threshold):
        denom = max(1.0 - threshold, 1e-12)

        norm_signal = max(0.0, min(1.0, (model_value - threshold) / denom))

        frac = self.config.tx_fraction + norm_signal * (
                    self.config.max_fraction_when_confident - self.config.tx_fraction)

        frac = max(0.0, min(1.0, frac))
        # frac mincost, qty
        return round(frac,4)

    def build_obs(self, symbols, data_rows, prob_vec, prices):

        obs_parts = [np.array([float(self.wallet.balance)], dtype=np.float32)]
        current_portfolio_value = self.wallet.total_value(self._get_current_prices())
        cash_ratio = self.wallet.balance / current_portfolio_value if current_portfolio_value > 0 else 0.0

        for symbol in symbols:
            buy_p = float(prob_vec[symbol].get("buy", 0.0))
            sell_p = float(prob_vec[symbol].get("sell", 0.0))
            hold_p = float(prob_vec[symbol].get("hold", 0.0))

            holding_amount = float(self.wallet.holdings.get(symbol, 0.0))
            current_price = float(data_rows[symbol].get("close", 0.0))
            holding_value = holding_amount * current_price
            position_ratio = holding_value / current_portfolio_value if current_portfolio_value > 0 else 0.0
            #numeric_values = np.asarray(data_rows[symbol], dtype=np.float32)
            # 3. AKTUALNY ZYSK/STRATA NA POZYCJI (Unrealized PnL %)
            # To jest kluczowe! Agent musi wiedzieć czy pozycja krwawi.
            avg_buy_price = self.wallet.avg_buy_price.get(symbol,0.0)  # Musisz to obsłużyć w Wallet (średnia cena zakupu)
            unrealized_pnl_pct = 0.0

            if holding_amount > 0 and avg_buy_price > 0:
                unrealized_pnl_pct = (current_price - avg_buy_price) / avg_buy_price
                # Clipujemy, żeby nie rozwalić sieci przy pompie o 1000%
                unrealized_pnl_pct = np.clip(unrealized_pnl_pct, -1.0, 1.0)
            #
            # Budujemy wektor dla danego symbolu
            symbol_obs = np.array([
                buy_p,
                hold_p,
                sell_p,
                position_ratio,  # Zamiast holding_amount
                unrealized_pnl_pct,  # Zamiast holding_value
                # rsi,  # Znormalizowany wskaźnik
                # log_ret_1m,  # Dynamika ceny
                # atr_pct  # Zmienność
            ], dtype=np.float32)

            obs_parts.append(symbol_obs)

            obs_parts.append(symbol_obs)

        obs = np.concatenate(obs_parts)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        return obs
