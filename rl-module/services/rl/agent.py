import gymnasium as gymnasium
from gymnasium import spaces
import numpy as np
from services.core_logic.wallet import Wallet, WalletConfig
from services.core_logic.decision import DecisionEngine, DecisionConfig

class TradingEnv(gymnasium.Env):
    def __init__(self, models, symbols,base_dir, initial_balance=1000.0):
        super().__init__()
        self.models = models
        self.symbols = symbols
        self.data = {}
        self.trades_count = 0
        self.core_mode = "train_rl"
        self.base_dir = base_dir
        self.prices = {}
        self.prev_prices = {}
        self.t = 0
        for symbol in self.symbols:
            df = models[symbol].data.reset_index(drop=True)
            self.data[symbol] = df
            self.prices[symbol] = df["close"].to_numpy()
            self.prev_prices[symbol] = df["close"][self.t]
        self.max_steps = min(len(self.data[s]) for s in self.symbols)
        self.episode_length = 1000


        self.initial_balance = float(initial_balance)
        self.prev_value = 0
        self.wallet = Wallet(balance=initial_balance, symbols=symbols,core_mode="train_rl", config=WalletConfig())
        self.engine = DecisionEngine(wallet=self.wallet, core_mode="train_rl",base_dir=base_dir, config=DecisionConfig())

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6 * len(symbols),), dtype=np.float32)
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

        self.wallet = Wallet(balance=self.initial_balance, symbols=self.symbols, core_mode=self.core_mode,
                             config=WalletConfig())
        self.engine = DecisionEngine(wallet=self.wallet, core_mode=self.core_mode, base_dir=self.base_dir,
                                     config=DecisionConfig())
        self.prev_value = self.initial_balance
        self.trades_count = 0
        # Inicjalizacja cen dla wylosowanego momentu t
        self.prev_prices = {}
        for symbol in self.symbols:
            df = self.models[symbol].data
            self.prev_prices[symbol] = df.iloc[self.t]["close"]

        return self._get_obs(), {}


    def _get_obs(self):
        prob_vec, close_prices= {},{}

        for symbol in self.symbols:
            prob_array = self.models[symbol].predictions[self.t]
            classes = self.models[symbol].classes
            prob_vec[symbol] = dict(zip(classes, prob_array))
            close_prices[symbol] = self.prices[symbol][self.t]

        obs = self.engine.build_obs(self.symbols, prob_vec, close_prices)
        return obs

    def reward(self):
        reward = 1.0
        return reward
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

        market_returns = []
        for symbol in self.symbols:
            prices[symbol] = self.models[symbol].data.iloc[self.t]["close"]
            data_rows[symbol] = self.models[symbol].data.iloc[self.t]
            prob_array = self.models[symbol].predictions[self.t]
            prob_vec[symbol] = dict(zip(self.models[symbol].classes, prob_array))

            # Obliczanie zmiany ceny dla konkretnego symbolu
            prev_p = self.prev_prices[symbol]
            curr_p = prices[symbol]
            if prev_p > 1e-6:
                # Wzór: (Nowa - Stara) / Stara
                ret = (curr_p - prev_p) / prev_p
                market_returns.append(ret)

        self.engine.generate_decisions(prob_vec, self.symbols, prices, self.t, scaled_dict)

        # 1. Obliczanie Agent Return
        portfolio_value = self.wallet.total_value(prices)
        prev_value = float(self.prev_value)
        #test_reward = portfolio_value - prev_value
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
        #reward = float(np.clip(test_reward, -10.0, 10.0))
        reward = np.log(portfolio_value / self.prev_value)
        reward *= 10.0
        # update prev_value i prev_prices (robimy kopię słownika prices!)
        self.prev_value = portfolio_value
        self.prev_prices = prices.copy()

        # postęp czasu
        self.t += 1
        symbol = self.symbols[0]
        terminated = (self.t >= self.end_step) or (self.t >= self.max_steps - 1)
        truncated = False
        #print(f"[DEBUG] terminated={terminated}" )
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
