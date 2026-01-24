import gymnasium as gymnasium
from gymnasium import spaces
import numpy as np
from ..core_logic.wallet import Wallet, WalletConfig
from ..core_logic.decision import DecisionEngine, DecisionConfig


class TradingEnv(gymnasium.Env):
    def __init__(self, models, symbols, num_ticks, base_dir, initial_balance=1000.0):
        super().__init__()
        self.models = models
        self.symbols = symbols
        self.trades_count = 0
        self.core_mode = "train_rl"
        self.base_dir = base_dir
        self.t = 0

        self.max_steps = num_ticks
        self.episode_length = 10240
        self.current_probs = {}
        self.current_prices = {}

        self.initial_balance = float(initial_balance)
        self.prev_value = 0
        self.wallet = Wallet(balance=self.initial_balance, symbols=self.symbols, core_mode=self.core_mode,
                             config=WalletConfig())
        self.engine = DecisionEngine(wallet=self.wallet, core_mode=self.core_mode, base_dir=self.base_dir,
                                     config=DecisionConfig())

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2 * len(symbols),), dtype=np.float32)
        obs_len = 1 + (5 * len(self.symbols))

        low = np.full(obs_len, -np.inf, dtype=np.float32)
        high = np.full(obs_len, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- ZMIANA: Losowy start (Random Slicing) ---
        # Obliczamy maksymalny możliwy indeks startowy
        max_start_index = self.max_steps - self.episode_length - 1

        if max_start_index > 0:
            self.t = np.random.randint(0, max_start_index)
        else:
            self.t = 0

        # Ustalamy, kiedy ten konkretny epizod ma się skończyć
        self.end_step = self.t + self.episode_length
        # ---------------------------------------------
        self.current_probs = {}
        self.current_prices = {}

        self.wallet.reset(self.initial_balance)
        self.engine.reset()

        self.prev_value = self.initial_balance
        self.trades_count = 0
        # Inicjalizacja cen dla wylosowanego momentu t


        return self._get_obs(), {}


    def _get_obs(self):
        volatility = {}
        for symbol in self.symbols:
            self.current_probs[symbol] = self.models[symbol].dict_probs[self.t]
            self.current_prices[symbol] = self.models[symbol].close_prices[self.t]
            volatility[symbol] = self.models[symbol].volatility[self.t]


        obs = self.engine.build_obs(self.current_probs, self.current_prices, volatility)
        return obs

    def calculate_reward(self):
        # 1. Obliczanie Agent Return
        portfolio_value = self.wallet.total_value(self.current_prices)
        prev_value = float(self.prev_value)
        # test_reward = portfolio_value - prev_value
        # agent_return = 0.0
        # if prev_value > 1e-6:
        #     agent_return = (portfolio_value - prev_value) / prev_value
        #
        # # 2. Obliczanie Market Return (średnia rynku)
        # market_return = 0.0
        # if market_returns:
        #     market_return = np.mean(market_returns)
        self.trades_count = self.wallet.transaction_count
        # # 3. Reward: Agent Return - Market Return (Alpha)
        # # Mnożymy x100, aby sieć "poczuła" różnicę
        # raw_reward = agent_return - market_return
        # reward = raw_reward * 1000.0
        #
        # # Clipowanie
        # reward = float(np.clip(test_reward, -10.0, 10.0))
        # reward = np.log(portfolio_value / self.prev_value)
        # reward *= 10.0
        # reward = round(reward,4)
        step_return = 0.0
        if self.prev_value > 0:
            step_return = np.log(portfolio_value / self.prev_value)

        # 2. Skalowanie (Scaling)
        # Log return dla 1 minuty to np. 0.0001. Sieć woli liczby bliższe 1.0.
        # Mnożymy przez 100 (lub 1000), żeby sygnał był wyraźny.
        scaled_return = step_return * 100.0

        # 3. Asymetria (Loss Aversion)
        # Jeśli tracimy, mnożymy karę x1.5. Agent ma bać się strat bardziej niż pożądać zysków.
        if scaled_return < 0:
            scaled_return *= 1.0  # .5 !!!!
        reward = scaled_return
        reward = np.clip(reward, -1, 1)
        # update prev_value i prev_prices (robimy kopię słownika prices!)
        self.prev_value = portfolio_value


        return reward
    def step(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()

        # skalowanie [-1,1] -> [0,1]
        scaled = (action + 1.0) / 2.0

        # mapowanie na słownik
        scaled_dict= {}
        n_params = 6
        for i, symbol in enumerate(self.symbols):
            idx = i * n_params
            scaled_dict[symbol] = scaled[idx: idx + n_params]
        market_returns = []
        # for symbol in self.symbols:
        #     # Obliczanie zmiany ceny dla konkretnego symbolu
        #     prev_p = self.prev_prices[symbol]
        #     curr_p = prices[symbol]
        #     if prev_p > 1e-6:
        #         # Wzór: (Nowa - Stara) / Stara
        #         ret = (curr_p - prev_p) / prev_p
        #         market_returns.append(ret)

        self.engine.generate_decisions(self.current_probs, self.current_prices, self.t, scaled=scaled_dict)
        self.wallet.update_history(self.current_prices)

        reward = self.calculate_reward()

        cum_reward = self.wallet.last_reward + reward
        self.wallet.rewards.append(round(cum_reward,4))
        self.wallet.last_reward = cum_reward
        # postęp czasu
        self.t += 1
        symbol = self.symbols[0]
        terminated = (self.t >= self.end_step) or (self.t >= self.max_steps - 1)
        truncated = False
        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}
        # for symbol in self.symbols:
        #     info[f"{symbol}_portfolio_value"] = float(portfolio_value)
        #     info[f"{symbol}_threshold_buy"] = float(scaled_dict[symbol][0])
        #     info[f"{symbol}_threshold_sell"] = float(scaled_dict[symbol][1])
        #     info[f"{symbol}_tx_fraction"] = float(scaled_dict[symbol][2])
        #     info[f"{symbol}_max_fraction_when_confident"] = float(scaled_dict[symbol][3])
        #     info[f"{symbol}_mincost"] = float(scaled_dict[symbol][4])
        #     info[f"{symbol}_max_wallet_percentage_crypto"] = float(scaled_dict[symbol][5])

        return obs, reward, terminated, truncated, info
