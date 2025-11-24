import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .rl_wallet import Wallet, WalletConfig
from .rl_decision import DecisionEngine, DecisionConfig

class TradingEnv(gym.Env):
    def __init__(self, models, symbols, initial_balance=1000.0, fee=0.001):
        super().__init__()
        self.models = models
        self.symbols = symbols
        self.data = {}
        self.initial_prices = {}
        self.numeric_columns = {}

        for symbol in self.symbols:
            df = models[symbol].data.reset_index(drop=True)
            self.data[symbol] = df
            self.initial_prices[symbol] = float(df.iloc[0]["close"])
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.numeric_columns[symbol] = numeric_cols

        self.max_steps = min(len(self.data[s]) for s in self.symbols)

        self.t = 0
        self.fee = float(fee)
        self.initial_balance = float(initial_balance)
        self.prev_value = self.initial_balance
        self.wallet = Wallet(balance=initial_balance, symbols=symbols, config=WalletConfig())
        self.engine = DecisionEngine(wallet=self.wallet, config=DecisionConfig())

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6 * len(symbols),), dtype=np.float32)
        obs_len = 1
        for symbol in self.symbols:
            obs_len += 3 + 2 + len(self.numeric_columns[symbol])

        low = np.full(obs_len, -np.inf, dtype=np.float32)
        high = np.full(obs_len, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.wallet = Wallet(balance=self.initial_balance, symbols=self.symbols, config=WalletConfig())
        self.engine = DecisionEngine(wallet=self.wallet, config=DecisionConfig())
        self.prev_value = self.initial_balance
        return self._get_obs(), {}

    def _get_obs(self):
        obs_parts = [np.array([float(self.wallet.balance)], dtype=np.float32)]

        for symbol in self.symbols:
            row = self.data[symbol].iloc[self.t]
            prob_array = self.models[symbol].predictions[self.t]
            classes = self.models[symbol].classes
            prob_map = dict(zip(classes, prob_array))

            buy_p = float(prob_map.get("buy", 0.0))
            sell_p = float(prob_map.get("sell", 0.0))
            hold_p = float(prob_map.get("hold", 0.0))

            holding_amount = float(self.wallet.holdings.get(symbol, 0.0))
            holding_value = holding_amount * float(row.get("close", 0.0))

            numeric_cols = self.numeric_columns[symbol]
            numeric_values = row[numeric_cols].fillna(0.0).to_numpy(dtype=np.float32, copy=False)

            symbol_obs = np.concatenate([
                np.array([buy_p, sell_p, hold_p], dtype=np.float32),
                np.array([holding_amount, holding_value], dtype=np.float32),
                numeric_values
            ])
            obs_parts.append(symbol_obs)

        obs = np.concatenate(obs_parts)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()

        # skalowanie [-1,1] -> [0,1]
        scaled = (action + 1.0) / 2.0

        # mapowanie na słownik
        scaled_dict, prices, data_rows, prob_vec = {}, {}, {}, {}
        n_params_per_symbol = 6
        for i, symbol in enumerate(self.symbols):
            start = i * n_params_per_symbol
            end = start + n_params_per_symbol
            scaled_dict[symbol] = scaled[start:end]

        for symbol in self.symbols:
            prices[symbol] = self.models[symbol].data.iloc[self.t]["close"]
            data_rows[symbol] = self.models[symbol].data.iloc[self.t]
            prob_array = self.models[symbol].predictions[self.t]
            prob_row = dict(zip(self.models[symbol].classes, prob_array))
            prob_vec[symbol] = prob_row

        self.engine.generate_decisions(prob_vec, self.symbols, prices, self.t, scaled_dict)

        # portfolio value i względny reward (normalizowany)
        portfolio_value = self.wallet.total_value(prices)
        # względna zmiana wartości względem poprzedniego kroku
        prev_value = float(self.prev_value)
        reward = 0.0
        if prev_value > 0:
            reward = (portfolio_value - prev_value) / prev_value

        reward = float(np.clip(reward, -1, 1))

        # update prev_value
        self.prev_value = portfolio_value

        # postęp czasu
        self.t += 1
        terminated = self.t >= len(self.data) - 1
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}
        for symbol in self.symbols:
            info[f"{symbol}_portfolio_value"] = float(portfolio_value)
            info[f"{symbol}_threshold_buy"] = float(scaled_dict[symbol][0])
            info[f"{symbol}_threshold_sell"] = float(scaled_dict[symbol][1])
            info[f"{symbol}_tx_fraction"] = float(scaled_dict[symbol][2])
            info[f"{symbol}_max_fraction_when_confident"] = float(scaled_dict[symbol][3])
            info[f"{symbol}_mincost"] = float(scaled_dict[symbol][4])
            info[f"{symbol}_max_wallet_percentage_crypto"] = float(scaled_dict[symbol][5])

        return obs, reward, terminated, truncated, info
