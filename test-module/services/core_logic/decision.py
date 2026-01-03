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
    def __init__(self, wallet, base_dir, core_mode, rlagent_deterministic=None, agent_name=None, config: DecisionConfig = DecisionConfig()):
        self.BASE_MODEL_DIR = os.path.join(base_dir, "shared", "agents")
        self.agent_name = agent_name
        self.agent_path = ""
        self.config = config
        self.core_mode = core_mode
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

        thr_low, thr_high = 0.1, 0.9
        threshold_buy = thr_low + scaled[0] * (thr_high - thr_low)
        threshold_sell = thr_low + scaled[1] * (thr_high - thr_low)

        tx_min, tx_max = 0.001, 1.0
        tx_fraction = tx_min + scaled[2] * (tx_max - tx_min)

        tx_min2, tx_max2 = 0.001, 1.0
        tx_max_fraction = tx_min2 + scaled[3] * (tx_max2 - tx_min2)
        self.config.threshold_buy = float(threshold_buy)
        self.config.threshold_sell = float(threshold_sell)
        self.config.tx_fraction = float(tx_fraction)
        self.config.max_fraction_when_confident = float(tx_max_fraction)

    def generate_decisions(self, prob_vec, symbols, prices, t, scaled = None):

        candidates = []

        if self.core_mode == "sim_rl":
            scaled = self.get_obs(symbols, prob_vec, prices)


        for symbol in symbols:
            # Pobieramy prawdopodobieństwa z modelu ML
            probs = prob_vec[symbol]
            buy_prob = float(probs.get("buy", 0.0))
            sell_prob = float(probs.get("sell", 0.0))

            if self.core_mode in ("train_rl", "sim_rl"):
                self.rl_config(scaled[symbol])

            # Domyślny stan
            action = "hold"
            value = 0.0
            fraction = 0.0
            reason = "hold"

            # Logika "Gatekeeper" - Agent RL decyduje, czy sygnał jest dość silny
            # Sprawdzamy SELL
            if sell_prob > self.config.threshold_sell:
                action = "sell"
                value = sell_prob
                fraction = self.calculate_fraction(value, self.config.threshold_sell)
                reason = "rl_signal_sell"

            # Sprawdzamy BUY (tylko jeśli nie sprzedajemy)
            elif buy_prob > self.config.threshold_buy:
                action = "buy"
                value = buy_prob
                fraction = self.calculate_fraction(value, self.config.threshold_buy)
                reason = "rl_signal_buy"

            # Jeśli Agent podjął decyzję inną niż HOLD, dodajemy do listy
            if action != "hold":
                candidates.append({
                    "symbol": symbol,
                    "action": action,  # "buy" lub "sell"
                    "value": value,  # Siła sygnału (np. 0.75)
                    "fraction": fraction,
                    "price": prices[symbol],
                    "timestamp": t,
                    "reason": reason,
                    "rl_config": scaled[symbol] if scaled is not None else "Brak" # Ważne dla Wallet do obliczenia mincost
                })

        # KROK 2: Sortowanie strategiczne (Klucz do "Sprzedaj A, by kupić B")

        # Rozdzielamy sprzedaże i kupna
        sells = [o for o in candidates if o['action'] == 'sell']
        buys = [o for o in candidates if o['action'] == 'buy']

        # Sortujemy wewnątrz grup malejąco po pewności (najpierw najpewniejsze ruchy)
        sells.sort(key=lambda x: x['value'], reverse=True)
        buys.sort(key=lambda x: x['value'], reverse=True)

        # Sklejamy listę: NAJPIERW SPRZEDAŻE, POTEM KUPNA
        # Dzięki temu Wallet najpierw uwolni USDT, a potem spróbuje je wydać.
        final_orders = sells + buys

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

    def get_obs(self, symbols, prob_vec, prices):
        scaled_dict = {}
        obs = self.build_obs(symbols, prob_vec, prices)
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

    def build_obs(self, symbols,  prob_vec, prices):

        obs_parts = [np.array([float(self.wallet.balance)], dtype=np.float32)]
        current_portfolio_value = self.wallet.total_value(prices)
        cash_ratio = self.wallet.balance / current_portfolio_value if current_portfolio_value > 0 else 0.0

        for symbol in symbols:
            buy_p = float(prob_vec[symbol].get("buy", 0.0))
            sell_p = float(prob_vec[symbol].get("sell", 0.0))
            hold_p = float(prob_vec[symbol].get("hold", 0.0))

            holding_amount = float(self.wallet.holdings.get(symbol, 0.0))
            current_price = float(prices[symbol])
            holding_value = holding_amount * current_price
            position_ratio = holding_value / current_portfolio_value if current_portfolio_value > 0 else 0.0
            avg_buy_price = self.wallet.avg_buy_price.get(symbol,0.0)  # Musisz to obsłużyć w Wallet (średnia cena zakupu)
            unrealized_pnl_pct = 0.0

            if holding_amount > 0 and avg_buy_price > 0:
                unrealized_pnl_pct = (current_price - avg_buy_price) / avg_buy_price
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

        obs = np.concatenate(obs_parts)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        return obs
