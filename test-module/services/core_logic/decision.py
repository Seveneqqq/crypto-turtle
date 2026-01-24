import os
import numpy as np
from dataclasses import dataclass
from stable_baselines3 import PPO, SAC
from .structures import Order

@dataclass
class DecisionConfig:
    threshold_buy: float = 0.6
    threshold_sell: float = 0.6
    tx_fraction: float = 0.1
    max_fraction_when_confident: float = 0.5
    clamp_model_value: bool = True

class DecisionEngine:
    def __init__(self, wallet, base_dir, core_mode = "sim_ml", rlagent_deterministic=None, agent_name=None, config: DecisionConfig = DecisionConfig()):
        self.BASE_MODEL_DIR = os.path.join(base_dir, "shared", "agents")
        self.agent_name = agent_name
        self.agent_path = ""
        self.config = config
        self.core_mode = core_mode
        self.wallet = wallet
        self.symbols = self.wallet.symbols
        self.rlagent_deterministic = rlagent_deterministic
        self.agent = None
        self.BUY_IDX = 0  # np. pobrane z class_map
        self.SELL_IDX = 2
        if self.agent_name:
            agent_type = self.check_agent_type()
            match agent_type:
                case "ppo":
                    self.agent = PPO.load(self.agent_path, env=None)
                case "sac":
                    self.agent = SAC.load(self.agent_path, env=None)
                case _:
                    raise ValueError(f"Unsupported agent type: {agent_type}")

    def reset(self):
        self.config = DecisionConfig()

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

        thr_low, thr_high = 0.45, 0.65
        threshold_buy = thr_low + scaled[0] * (thr_high - thr_low)
        threshold_sell = thr_low + scaled[1] * (thr_high - thr_low)

        # tx_min, tx_max = 0.01, 1.0
        # tx_fraction = tx_min + scaled[2] * (tx_max - tx_min)
        #
        # tx_min2, tx_max2 = 0.45, 1.0
        # tx_max_fraction = tx_min2 + scaled[3] * (tx_max2 - tx_min2)

        self.config.threshold_buy = float(round(threshold_buy,2))
        self.config.threshold_sell = float(round(threshold_sell,2))
        # self.config.tx_fraction = float(round(tx_fraction,2))
        # self.config.max_fraction_when_confident = float(round(tx_max_fraction,2))

    def generate_decisions(self, prob_vec, prices, t, volatility=None, scaled = None):

        candidates = []

        if self.core_mode == "sim_rl":
            scaled = self.get_obs(prob_vec, prices, volatility)


        for symbol in self.symbols:
            # Pobieramy prawdopodobieństwa z modelu ML
            probs = prob_vec[symbol]  # To teraz jest wiersz numpy array, np. [0.1, 0.8, 0.1]
            buy_prob = probs[self.BUY_IDX]
            sell_prob = probs[self.SELL_IDX]

            if self.core_mode in ("train_rl", "sim_rl"):
                self.rl_config(scaled[symbol])

            buy_diff_prob = buy_prob - self.config.threshold_buy
            sell_diff_prob = sell_prob - self.config.threshold_sell
            # Logika "Gatekeeper" - Agent RL decyduje, czy sygnał jest dość silny

            if sell_diff_prob >= buy_diff_prob and sell_prob >= self.config.threshold_sell:
                action = "sell"
                value = sell_prob
                fraction = self.calculate_fraction(value, self.config.threshold_sell)
            elif buy_diff_prob > sell_diff_prob and buy_prob >= self.config.threshold_buy:
                action = "buy"
                value = buy_prob
                fraction = self.calculate_fraction(value, self.config.threshold_buy)
            else:
                action = "hold"
                value = 0.0
                fraction = 0.0

            order = Order(
                timestamp=t,
                symbol=symbol,
                action=action,
                value=value,
                fraction=fraction,
                price=prices[symbol],
                rl_config=scaled[symbol] if scaled is not None else "Brak"
            )
            if order.action == "hold": order.rejected("lower_than_thresholds")

            candidates.append(order)


        sells = [o for o in candidates if o.action == 'sell']
        buys = [o for o in candidates if o.action == 'buy']
        holds = [o for o in candidates if o.action == 'hold']

        # Sortujemy wewnątrz grup malejąco po pewności (najpierw najpewniejsze ruchy)
        sells.sort(key=lambda x: x.value, reverse=True)
        buys.sort(key=lambda x: x.value, reverse=True)

        final_orders = sells + buys
        self.wallet.logs.extend(holds)
        # KROK 3: Wysłanie posortowanej paczki do portfela
        self.wallet.orders.extend(final_orders)
        self.wallet.execute_orders(t, prices)

    def calculate_fraction(self, model_value, threshold):
        denom = max(1.0 - threshold, 1e-12)

        norm_signal = max(0.0, min(1.0, (model_value - threshold) / denom))

        frac = self.config.tx_fraction + norm_signal * (
                    self.config.max_fraction_when_confident - self.config.tx_fraction)

        frac = max(0.0, min(1.0, frac))
        # frac mincost, qty
        return round(frac,4)

    def get_obs(self, prob_vec, prices, volatility):
        scaled_dict = {}

        obs = self.build_obs(prob_vec, prices, volatility)
        action, _ = self.agent.predict(obs, deterministic=self.rlagent_deterministic)
        action = np.asarray(action, dtype=np.float32).flatten()
        scaled = (action + 1.0) / 2.0
        #print(scaled)
        n_params_per_symbol = 2
        for i, symbol in enumerate(self.symbols):
            start = i * n_params_per_symbol
            end = start + n_params_per_symbol
            scaled_dict[symbol] = scaled[start:end]
        return scaled_dict


    def build_obs(self, prob_vec, prices, volatility):

        current_portfolio_value = self.wallet.total_value(prices)
        cash_ratio = self.wallet.balance / current_portfolio_value if current_portfolio_value > 0 else 0.0

        obs_parts = [np.array([float(cash_ratio)], dtype=np.float32)]

        for symbol in self.symbols:

            buy_p = prob_vec[symbol][self.BUY_IDX]
            sell_p = prob_vec[symbol][self.SELL_IDX]
            #hold_p = float(prob_vec[symbol].get("hold", 0.0))

            holding_amount = float(self.wallet.holdings.get(symbol, 0.0))
            current_price = float(prices[symbol])
            holding_value = holding_amount * current_price
            position_ratio = holding_value / current_portfolio_value if current_portfolio_value > 0 else 0.0
            avg_buy_price = self.wallet.avg_buy_price.get(symbol,0.0)  # Musisz to obsłużyć w Wallet (średnia cena zakupu)
            unrealized_pnl_pct = 0.0

            if holding_amount > 0 and avg_buy_price > 0:
                unrealized_pnl_pct = (current_price - avg_buy_price) / avg_buy_price
                #unrealized_pnl_pct = np.clip(unrealized_pnl_pct, -1.0, 1.0)

            symbol_obs = np.array([
                buy_p,
                #hold_p,
                sell_p,
                volatility[symbol],
                position_ratio,  # Zamiast holding_amount
                unrealized_pnl_pct,  # Zamiast holding_value
            ], dtype=np.float32)

            obs_parts.append(symbol_obs)

        obs = np.concatenate(obs_parts)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        obs = np.clip(obs, -1.0, 1.0)
        return obs
