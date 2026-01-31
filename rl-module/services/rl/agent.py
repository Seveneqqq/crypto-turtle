import gymnasium as gymnasium
from gymnasium import spaces
import numpy as np
from ..core_logic.decision import DecisionEngine
from ..core_logic.structures import PreviousEpoch,  CoreConfig


class TradingEnv(gymnasium.Env):
    def __init__(self, models, num_ticks, base_dir, core_config: CoreConfig = None):
        super().__init__()
        self.models = models
        self.symbols = core_config.symbols
        self.core_mode = "train_rl"
        self.base_dir = base_dir
        self.t = 0
        self.end_step = 0

        self.prev_epoch = PreviousEpoch()

        self.max_steps = num_ticks
        self.episode_length = 10240

        self.current_prices = {}
        self.prev_prices = {}
        self.initial_balance = core_config.balance_start
        self.prev_value = 0
        self.current_value = 0

        # Callback Counters

        self.last_reward = 0
        self.rewards = []
        self.epoch_counter = 0

        self.engine = DecisionEngine(core_mode=self.core_mode,models=self.models, base_dir=self.base_dir,
                                     core_config=core_config)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2 * len(self.symbols),), dtype=np.float32)

        obs_len = 1 + (9 * len(self.symbols))
        low = np.full(obs_len, -np.inf, dtype=np.float32)
        high = np.full(obs_len, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.reset()

    def prepare_callback(self):
        if self.engine.wallet.transaction_count > 0:
            self.prev_epoch.buys = self.engine.wallet.buys_count
            self.prev_epoch.sells = self.engine.wallet.sells_count
            self.prev_epoch.transactions = self.engine.wallet.transaction_count
            self.prev_epoch.portfolio_values = self.engine.wallet.portfolio_values_history
            self.prev_epoch.rewards = self.rewards
            self.prev_epoch.calculate_history()
            self.epoch_counter += 1
            self.prev_epoch.epoch = self.epoch_counter


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # -----------New epoch

        max_start_index = self.max_steps - self.episode_length - 1
        if max_start_index > 0:
            self.t = np.random.randint(0, max_start_index)
        else:
            self.t = 0

        self.end_step = self.t + self.episode_length

        # ------------Reset Env
        self.prev_value = self.initial_balance
        self.current_value = self.initial_balance


        self.current_prices = {}
        self.prev_prices = {}

        self.prepare_callback()
        self.engine.wallet.reset(self.initial_balance)
        self.engine.reset()
        self.last_reward = 0
        self.rewards = []

        return self._get_obs(), {}


    def _get_obs(self):
        obs = self.engine.build_obs(self.t)
        return obs



    #TODO: Pobieranie danych w jednym miejscu
    def prepare_data(self):

        self.prev_prices = self.current_prices

        for symbol in self.symbols:

            self.current_prices[symbol] = self.models[symbol].close_prices[self.t]

        self.current_value = self.engine.wallet.total_value(self.current_prices)



    def _calculate_reward(self):
        """
        Oblicza nagrodę na podstawie Alpha (Agent Return --- Market Return)
        oraz karze za obsunięcia kapitału (Sortino-like).
        """
        # 1. Agent Return (Log Return)
        agent_return = 0.0

        if self.prev_value > 0:
            agent_return = np.log(self.current_value / self.prev_value)



        # 2. Market Return (Benchmark - średnia log returnów coinów)
        market_returns = []
        for symbol in self.symbols:
            p_curr = self.current_prices[symbol]
            p_prev = self.prev_prices.get(symbol, p_curr)
            if p_prev > 1e-9:
                market_returns.append(np.log(p_curr / p_prev))

        market_return = np.mean(market_returns) if market_returns else 0.0

        # 3. Alpha
        alpha = agent_return - market_return

        # 4. Skalowanie
        reward = alpha * 10.0
        if agent_return < 0:
            reward += (agent_return * 10.0)

        # 5. Kary (Asymetria)
        #Kara za stratę absolutną (Loss Aversion) - agent ma unikać strat bardziej niż szukać zysku
        if market_return > 0.005 and abs(agent_return) < 0.0001:
            reward -= 0.1

        # 6. Clipping (żeby gradient nie wybuchł)
        # reward = np.clip(reward, -10.0, 10.0)

        return agent_return#, info

    def step(self, action):

        action = np.asarray(action, dtype=np.float32).flatten()
        scaled = (action + 1.0) / 2.0
        # mapowanie na słownik
        scaled_dict= {}
        n_params = 6
        for i, symbol in enumerate(self.symbols):
            idx = i * n_params
            scaled_dict[symbol] = scaled[idx: idx + n_params]


        self.engine.generate_decisions(self.t, scaled=scaled_dict)
        self.engine.wallet.update_history()
        self.t += 1
        # Pobranie NOWYCH danych rynkowych (dla t+1)

        self.prepare_data()

        # 5. OBLICZENIE NAGRODY (Wydzielona funkcja)
        reward = self._calculate_reward()

        self.prev_value = self.current_value
        #--------------Portfolio previous-----------------

        self.engine.wallet.portfolio_values_history.append(self.current_value)


        #--------------Rewardy previous-------------------

        cum_reward = self.last_reward + reward
        self.rewards.append(round(cum_reward, 4))
        self.last_reward = cum_reward

        # postęp czasu


        terminated = (self.t >= self.end_step) or (self.t >= self.max_steps - 1)
        truncated = False
        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        observation = obs.copy()
        info = {}
        # for symbol in self.symbols:
        #     info[f"{symbol}_portfolio_value"] = float(portfolio_value)
        #     info[f"{symbol}_threshold_buy"] = float(scaled_dict[symbol][0])
        #     info[f"{symbol}_threshold_sell"] = float(scaled_dict[symbol][1])
        #     info[f"{symbol}_tx_fraction"] = float(scaled_dict[symbol][2])
        #     info[f"{symbol}_max_fraction_when_confident"] = float(scaled_dict[symbol][3])
        #     info[f"{symbol}_mincost"] = float(scaled_dict[symbol][4])
        #     info[f"{symbol}_max_wallet_percentage_crypto"] = float(scaled_dict[symbol][5])

        return observation, reward, terminated, truncated, info

