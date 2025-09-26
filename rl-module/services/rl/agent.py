import gym
import numpy as np
from gym import spaces
from services.rl.wallet import Wallet
from services.rl.decision import DecisionEngine

class TradingEnv(gym.Env):
    """
    Środowisko RL dla handlu.
    Agent uczy się optymalizować progi DecisionEngine (threshold_buy/sell)
    korzystając z portfela Wallet.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, data, model_path, symbol, initial_balance=1000.0,
                 wallet_percentage_limit=0.1, mincost=10.0):
        super(TradingEnv, self).__init__()

        self.data = data
        self.model_path = model_path
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.current_step = 0

        # --- komponenty ---
        self.engine = DecisionEngine(model_path)
        self.wallet = Wallet(initial_balance, symbols=[symbol],
                             threshold_buy=0.6, threshold_sell=0.6,
                             wallet_percentage_limit=wallet_percentage_limit,
                             mincost=mincost)

        # feature names jak w modelu (jeśli będziesz chciał użyć danych raw zamiast prob_row)
        self.FEATURE_COLUMNS = self.engine.model.feature_names_in_

        # --- action space ---
        # Agent steruje progami [buy, sell] -> wartości od 0.5 do 0.9
        self.action_space = spaces.Box(low=np.array([0.5, 0.5]),
                                       high=np.array([0.9, 0.9]),
                                       dtype=np.float32)

        # --- observation space ---
        # probabilistyka (buy/sell/hold) + balans + holdings
        obs_size = 3 + 2
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(obs_size,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.wallet = Wallet(self.initial_balance, symbols=[self.symbol],
                             threshold_buy=0.6, threshold_sell=0.6,
                             wallet_percentage_limit=self.wallet.wallet_percentage_limit,
                             mincost=self.wallet.mincost)
        return self._get_observation()

    def step(self, action):
        # ustalenie progów na podstawie akcji agenta
        self.engine.threshold_buy = float(action[0])
        self.engine.threshold_sell = float(action[1])

        row = self.data.iloc[self.current_step]

        # zakładam, że row zawiera już kolumny ["buy","sell","hold"] z predict_proba
        prob_row = row[["buy", "sell", "hold"]]

        model_value, decision = self.engine.decide(prob_row)

        # wykonanie transakcji
        executed_action, _ = self.wallet.execute(decision, self.symbol, row["close"], model_value)

        # nagroda = aktualna wartość portfela
        reward = self.wallet.total_value({self.symbol: row["close"]})

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = self._get_observation()

        return obs, reward, done, {}

    def _get_observation(self):
        row = self.data.iloc[self.current_step]
        return np.array([
            row["buy"],
            row["sell"],
            row["hold"],
            self.wallet.balance,
            self.wallet.holdings[self.symbol]
        ], dtype=np.float32)

    def render(self, mode="human"):
        print(f"Step {self.current_step}, Balance={self.wallet.balance:.2f}, Holdings={self.wallet.holdings[self.symbol]:.6f}")
