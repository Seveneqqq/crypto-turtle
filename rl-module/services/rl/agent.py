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

    def __init__(self, data, model_path, initial_balance=1000.0):
        super(TradingEnv, self).__init__()

        self.data = data
        self.model_path = model_path
        self.initial_balance = initial_balance
        self.current_step = 0

        # --- komponenty ---
        self.engine = DecisionEngine(model_path)
        self.wallet = Wallet(initial_balance, symbols=["BTC"], threshold_buy=0.6, threshold_sell=0.6)

        # --- action space ---
        # Agent steruje progami [buy, sell] -> wartości od 0.5 do 0.9
        self.action_space = spaces.Box(low=np.array([0.5, 0.5]), high=np.array([0.9, 0.9]), dtype=np.float32)

        # --- observation space ---
        # close, return, ma_5, ma_15, balance, holdings
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.wallet = Wallet(self.initial_balance, symbols=["BTC"], threshold_buy=0.6, threshold_sell=0.6)
        return self._get_observation()

    def step(self, action):
        # ustalenie progów na podstawie akcji agenta
        self.engine.threshold_buy = float(action[0])
        self.engine.threshold_sell = float(action[1])

        row = self.data.iloc[self.current_step]
        features = row[["close", "return", "ma_5", "ma_15"]]
        model_value, decision = self.engine.decide(features)

        # wykonanie transakcji
        executed_action, _ = self.wallet.execute(decision, "BTC", row["close"], model_value)

        # nagroda = aktualna wartość portfela
        reward = self.wallet.total_value({"BTC": row["close"]})

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = self._get_observation()

        return obs, reward, done, {}

    def _get_observation(self):
        row = self.data.iloc[self.current_step]
        return np.array([
            row["close"],
            row["return"],
            row["ma_5"],
            row["ma_15"],
            self.wallet.balance,
            self.wallet.holdings["BTC"]
        ], dtype=np.float32)

    def render(self, mode="human"):
        print(f"Step {self.current_step}, Balance={self.wallet.balance:.2f}, Holdings={self.wallet.holdings['BTC']:.6f}")
