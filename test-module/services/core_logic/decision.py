import os
import numpy as np
import types
from stable_baselines3 import PPO, SAC
from .version_registry import EXPERIMENTS
from .structures import Order, DecisionConfig, CoreConfig, WalletConfig
from .wallet import Wallet

class DecisionEngine:
    def __init__(self, base_dir, core_mode = "sim_ml", models=None, agent_name=None, core_config: CoreConfig = None):
        self.BASE_MODEL_DIR = os.path.join(base_dir, "shared", "agents")
        self.agent_name = agent_name
        self.agent_path = ""
        self.core_mode = core_mode
        self.core_config = core_config

        self.symbols = self.core_config.symbols
        self.config = {}
        for symbol in self.symbols:
            self.config[symbol] = DecisionConfig()
            if core_mode in ("sim_rl","sim_ml"):  self.config[symbol].set(core_config)

        self.models = models

        self.current_prices = {}
        self.current_probs = {}
        self.current_value = 0
        self.prev_value = 0
        self.experiment_version = core_config.experiment_version
        self.n_params_per_symbol = 0
        self.wallet = Wallet(self.core_mode, core_config)

        self.rlagent_deterministic = core_config.rlagent_deterministic
        self.agent = None
        self.BUY_IDX = 0  #TODO: np. pobrane z class_map
        self.HOLD_IDX = 1
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
        self.apply_experiment_version()

    def init_local_configs(self, symbol):
        self.config[symbol] = DecisionConfig()
        self.config[symbol].set(self.core_config)
        self.wallet.config[symbol] = WalletConfig()
        self.wallet.config[symbol].set(self.core_config)

    def reset(self):
        self.config = {}
        for symbol in self.symbols:
            self.config[symbol] = DecisionConfig()
            if self.core_mode in ("sim_rl", "sim_ml"):  self.config[symbol].set(self.core_config)
        self.prev_value = self.core_config.balance_start
        self.current_value = self.core_config.balance_start

    def apply_experiment_version(self):
        """Binduje custom methods z experiment registry"""


        if self.experiment_version not in EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment: {self.experiment_version}. "
                f"Available: {list(EXPERIMENTS.keys())}"
            )

        exp = EXPERIMENTS[self.experiment_version]

        # Override n_params_per_symbol
        self.n_params_per_symbol = exp["n_params_per_symbol"]

        # Bind Decision methods
        for method_name, method_func in exp["decision_methods"].items():
            bound_method = types.MethodType(method_func, self)
            setattr(self, method_name, bound_method)

        # Bind Wallet methods
        for method_name, method_func in exp["wallet_methods"].items():
            bound_method = types.MethodType(method_func, self.wallet)
            setattr(self.wallet, method_name, bound_method)

        # print(f"✓ Applied experiment version: {self.experiment_version}")
        # print(f"  - n_params_per_symbol: {self.n_params_per_symbol}")
        # print(f"  - Decision methods: {list(exp['decision_methods'].keys())}")
        # print(f"  - Wallet methods: {list(exp['wallet_methods'].keys())}")

    # def _normalize_model_value(self, mv):
    #     if mv is None:
    #         return 0.0
    #     mv = float(mv)
    #     if self.config["ETHUSDT"].clamp_model_value:
    #         mv = max(0.0, min(1.0, mv))
    #     return mv

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

    def rl_config(self, scaled_dict):
        for symbol in self.symbols:
            scaled = scaled_dict[symbol]
            thr_low, thr_high = 0.25, 0.85
            threshold_buy = thr_low + scaled[0] * (thr_high - thr_low)
            threshold_sell = thr_low + scaled[1] * (thr_high - thr_low)

            tx_min, tx_max = 0.01, 1.0
            tx_fraction = tx_min + scaled[2] * (tx_max - tx_min)

            tx_min2, tx_max2 = 0.45, 1.0
            tx_max_fraction = tx_min2 + scaled[3] * (tx_max2 - tx_min2)

            self.config[symbol].threshold_buy = float(round(threshold_buy,2))
            self.config[symbol].threshold_sell = float(round(threshold_sell,2))
            self.config[symbol].tx_fraction = float(round(tx_fraction,2))
            self.config[symbol].max_fraction_when_confident = float(round(tx_max_fraction,2))

    def t_data(self, t):
        for symbol in self.symbols:
            self.current_probs[symbol] = self.models[symbol].predictions[t]
            self.current_prices[symbol] = self.models[symbol].close_prices[t]

        self.prev_value = self.current_value
        self.current_value = self.wallet.total_value(self.current_prices)
        pass

    def generate_decisions(self, t, scaled = None):
        if self.core_mode in ("sim_rl","sim_ml"):
            self.t_data(t)

        candidates = []

        if self.core_mode == "sim_rl":
            scaled = self.get_obs(t)

        if self.core_mode in ("train_rl", "sim_rl"):
            self.rl_config(scaled)

        for symbol in self.symbols:
            # Pobieramy prawdopodobieństwa z modelu ML
            probs = self.current_probs[symbol]
            buy_prob = probs[self.BUY_IDX]
            sell_prob = probs[self.SELL_IDX]


            buy_diff_prob = buy_prob - self.config[symbol].threshold_buy
            sell_diff_prob = sell_prob - self.config[symbol].threshold_sell

            if sell_diff_prob >= buy_diff_prob and sell_prob >= self.config[symbol].threshold_sell:
                action = "sell"
                value = sell_prob
            elif buy_diff_prob > sell_diff_prob and buy_prob >= self.config[symbol].threshold_buy:
                action = "buy"
                value = buy_prob
            else:
                action = "hold"
                value = 0.0
                fraction = 0.0

            order = Order(
                timestamp = t,
                symbol = symbol,
                action = action,
                value  = round(value,4),
                buy_p  = round(buy_prob,4),
                hold_p = round(probs[self.HOLD_IDX],4) if self.HOLD_IDX else 0.0,
                sell_p = round(sell_prob,4),
                price  = self.current_prices[symbol],
                rl_config = scaled[symbol] if scaled is not None else "-",
            )
            if order.action == "hold":
                order.rejected("lower_than_thresholds")
            else:
                order = self.calculate_fraction(order)
            candidates.append(order)

        sells = [o for o in candidates if o.action == 'sell']
        buys =  [o for o in candidates if o.action == 'buy']
        holds = [o for o in candidates if o.action == 'hold']

        # Sortujemy wewnątrz grup malejąco po pewności (najpierw najpewniejsze ruchy)
        sells.sort(key=lambda x: x.value, reverse=True)
        buys.sort(key=lambda x: x.value, reverse=True)

        final_orders = sells + buys + holds
        self.wallet.orders.extend(final_orders)
        self.wallet.execute_orders(t, self.current_prices)

    def calculate_fraction(self, order):
        symbol = order.symbol
        if order.action == "buy":
              threshold = self.config[symbol].threshold_buy
        else: threshold = self.config[symbol].threshold_sell

        denom = max(1.0 - threshold, 1e-12)
        norm_signal = max(0.0, min(1.0, (order.value - threshold) / denom))

        frac = self.config[symbol].tx_fraction + norm_signal * (self.config[symbol].max_fraction_when_confident - self.config[symbol].tx_fraction)
        frac = max(0.0, min(1.0, frac))
        order.fraction = round(frac,4)

        return order

    def get_obs(self, t):
        scaled_dict = {}

        obs = self.build_obs(t)
        action, _ = self.agent.predict(obs, deterministic=self.rlagent_deterministic)
        action = np.asarray(action, dtype=np.float32).flatten()

        scaled = (action + 1.0) / 2.0

        for i, symbol in enumerate(self.symbols):
            start = i * self.n_params_per_symbol
            end = start + self.n_params_per_symbol
            scaled_dict[symbol] = scaled[start:end]
        return scaled_dict

    def build_obs(self, t):
        self.t_data(t)

        # 1. Globalne portfolio (bez zmian, jest OK)
        current_portfolio_value = self.wallet.total_value(self.current_prices)
        cash_ratio = self.wallet.balance / current_portfolio_value if current_portfolio_value > 0 else 0.0
        obs_parts = [np.array([float(cash_ratio)], dtype=np.float32)]

        for symbol in self.symbols:
            # A. Pewność modelu ML (Entropy)
            # To kluczowe dla Gatekeepera! Jeśli model ML waha się (0.4 buy, 0.4 sell),
            # to entropia jest wysoka -> RL powinien podnieść progi (nie handlować).
            probs = self.current_probs[symbol]
            buy_p = probs[self.BUY_IDX]
            sell_p = probs[self.SELL_IDX]

            # B. Zmienność rynku (Log Returns + Volatility)
            # Zamiast różnicy cen, weź logarytmiczną stopę zwrotu (znormalizowaną)
            price_t = self.models[symbol].close[t]
            price_t_minus_1 = self.models[symbol].close[t - 1] if t > 0 else price_t

            # Log return: niezależny od rzędu wielkości (działa tak samo dla BTC i SHIB)
            log_ret = np.log(price_t / price_t_minus_1) if price_t_minus_1 > 0 else 0.0

            # "Skalowanie" dla sieci neuronowej (log returny są małe, np. 0.001, mnożymy x100)
            market_momentum = log_ret * 100.0

            # C. Pozycja i PnL (bez zmian, jest OK)
            holding_amount = float(self.wallet.holdings.get(symbol, 0.0))
            holding_value = holding_amount * float(self.current_prices[symbol])
            position_ratio = holding_value / current_portfolio_value if current_portfolio_value > 0 else 0.0

            # Zamiast surowego PnL, który może być > 1000%, spłaszczamy go funkcją tanh
            avg_buy_price = self.wallet.avg_buy_price.get(symbol, 0.0)
            unrealized_pnl = 0.0
            if holding_amount > 0 and avg_buy_price > 0:
                raw_pnl = (self.current_prices[symbol] - avg_buy_price) / avg_buy_price
                unrealized_pnl = np.tanh(raw_pnl)  # Zawsze w zakresie [-1, 1]

            symbol_obs = np.array([
                buy_p,  # Co mówi ML?
                sell_p,  # Co mówi ML?
                # self.models[symbol].volatility[t],
                market_momentum,  # Czy rynek zapierdziela? (zastępuje różnicę cen)
                position_ratio,  # Ile tego mamy?
                unrealized_pnl  # Czy zarabiamy na tym?
            ], dtype=np.float32)

            obs_parts.append(symbol_obs)

        # Składamy i przycinamy
        obs = np.concatenate(obs_parts)
        return np.clip(obs, -5.0, 5.0)