import gymnasium as gymnasium
from gymnasium import spaces
import numpy as np
import types
from ..core_logic_v2.decision import DecisionEngine
from ..core_logic_v2.structures import PreviousEpoch,  CoreConfig
from ..core_logic_v2.version_registry import EXPERIMENTS

class TradingEnv(gymnasium.Env):
    def __init__(self, models, num_ticks, base_dir, core_config: CoreConfig = None, episode_timesteps = 10240, agent_type=None):
        super().__init__()

        #TODO: przystosowanie pod dqn

        self.symbols = core_config.symbols
        self.core_mode = "train_rl"
        self.base_dir = base_dir
        self.experiment_version = core_config.experiment_version

        # Iteration and callback config
        self.t = 0
        self.end_step = 0
        self.max_steps = num_ticks
        self.episode_length = episode_timesteps
        self.epoch_counter = 0
        self.action_space_shape = 0
        self.prev_epoch = PreviousEpoch()
        self.last_reward = 0
        self.rewards = []
        self.initial_balance = core_config.balance_start
        self._apply_experiment_version()

        self.engine = DecisionEngine(core_mode=self.core_mode,models=models, base_dir=self.base_dir,
                                     core_config=core_config)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_space_shape * len(self.symbols),), dtype=np.float32)

        obs_len = 1 + (5 * len(self.symbols))
        low = np.full(obs_len, -np.inf, dtype=np.float32)
        high = np.full(obs_len, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.reset()

    def _apply_experiment_version(self):
        """Binduje custom methods z experiment registry"""


        if self.experiment_version not in EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment: {self.experiment_version}. "
                f"Available: {list(EXPERIMENTS.keys())}"
            )

        exp = EXPERIMENTS[self.experiment_version]

        # Override n_params_per_symbol
        self.action_space_shape = exp["n_params_per_symbol"]

        # Bind Decision methods
        for method_name, method_func in exp["env_methods"].items():
            bound_method = types.MethodType(method_func, self)
            setattr(self, method_name, bound_method)


        # print(f"✓ Applied experiment version: {self.experiment_version}")
        # print(f"  - n_params_per_symbol: {self.n_params_per_symbol}")
        # print(f"  - Decision methods: {list(exp['decision_methods'].keys())}")
        # print(f"  - Wallet methods: {list(exp['wallet_methods'].keys())}")

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

        max_start_index = self.max_steps - self.episode_length - 1
        if max_start_index > 0:
            self.t = np.random.randint(0, max_start_index)
        else:
            self.t = 0

        self.end_step = self.t + self.episode_length

        # ------------Reset Env

        self.prepare_callback()
        self.engine.wallet.reset(self.initial_balance)
        self.engine.reset()
        self.last_reward = 0
        self.rewards = []
        self.engine.t_data(self.t)
        return self._get_obs(), {}

    def _get_obs(self):
        obs = self.engine.build_obs(self.t)
        return obs

    def _calculate_reward(self):
        """
        Oblicza nagrodę na podstawie Alpha (Agent Return --- Market Return)
        oraz karze za obsunięcia kapitału (Sortino-like).
        """
        # 1. Logarytmiczna stopa zwrotu PORTFELA (nie ceny)
        # Jeśli portfel wzrósł z 1000$ na 1010$, to świetnie.
        portfolio_return = 0.0
        if self.engine.prev_value > 0:
            portfolio_return = np.log(self.engine.current_value / self.engine.prev_value)

            # 2. Skalowanie (żeby liczby nie były rzędu 0.00001)
            # Mnożymy razy 100, żeby procenty były jednościami
        reward = portfolio_return * 100.0
        # if self.prev_value == self.current_value:
        #     reward -= 0.01
        # if self.prev_value < self.current_value:
        #     reward += 0.05
        # 3. Kara za drawdown (obsunięcie)
        # Jeśli tracimy, kara jest podwójna. To uczy agenta ostrożności.
        if reward < 0:
            reward *= 2  # Pain is stronger than pleasure

        # 4. (Opcjonalnie) Mała kara za każdą transakcję (wymusza jakość, nie ilość)
        # Wymaga dostępu do licznika transakcji w tym kroku
        # if self.engine.wallet.transactions_in_this_step > 0:
        #     reward -= 0.05
        # if self.current_value > 1000.0:
        #     reward += (self.current_value -1000) / 10000

        return reward

    # def _calculate_reward2(self):
    #     # --- Agent's actual return ---
    #     portfolio_return = 0.0
    #     if self.engine.prev_value > 0:
    #         portfolio_return = np.log(self.engine.current_value / self.engine.prev_value)
    #
    #     # --- Benchmark: equal-weight buy & hold ---
    #     # (oblicz raz przy resecie, cached)
    #     benchmark_return = 0.0
    #     n = len(self.symbols)
    #     for symbol in self.symbols:
    #         prices = self.engine.models[symbol].close_prices
    #         if self.t > 0 and prices[self.t - 1] > 0:
    #             benchmark_return += np.log(prices[self.t] / prices[self.t - 1]) / n
    #
    #     # --- Alpha: ile lepiej od rynku? ---
    #     alpha = (portfolio_return - benchmark_return) * 100.0
    #
    #     # Kara za drawdown tylko na alpha, nie na rynek
    #     if alpha < 0:
    #         alpha *= 1.5
    #
    #     # Mała kara za transakcję (wymusza jakość sygnałów)
    #     # Sprawdź ile orderów zostało accepted w tym stepie
    #     accepted_this_step = sum(
    #         1 for o in self.engine.wallet.logs
    #         if o.timestamp == self.t - 1 and o.status == "accepted"
    #     )
    #     transaction_penalty = accepted_this_step * 0.02
    #
    #     return alpha - transaction_penalty
    def step(self, action):

        action = np.asarray(action, dtype=np.float32).flatten()
        scaled = (action + 1.0) / 2.0
        # mapowanie na słownik
        scaled_dict= {}
        n_params = 4
        for i, symbol in enumerate(self.symbols):
            idx = i * n_params
            scaled_dict[symbol] = scaled[idx: idx + n_params]


        self.engine.generate_decisions(self.t, scaled=scaled_dict)
        self.t += 1
        # Pobranie NOWYCH danych rynkowych (dla t+1)

        self.engine.t_data(self.t)

        # 5. OBLICZENIE NAGRODY (Wydzielona funkcja)
        reward = self._calculate_reward()

        self.engine.wallet.portfolio_values_history.append(self.engine.current_value)

        cum_reward = self.last_reward + reward
        self.rewards.append(round(cum_reward, 4))
        self.last_reward = cum_reward

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

