import gymnasium as gym
from gymnasium import spaces
import numpy as np
from services.rl.wallet import Wallet, WalletConfig
from services.rl.decision import DecisionEngine, DecisionConfig

class TradingEnv(gym.Env):
    def __init__(self, data, model, initial_balance=1000.0, fee=0.001):
        super().__init__()
        self.data = data
        self.model = model
        self.fee = fee
        self.initial_balance = initial_balance

        # wybór symbolu ceny
        if "close" in data.columns:
            self.symbol = "close"
        else:
            num_cols = data.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                raise ValueError("Brak kolumn numerycznych w danych.")
            self.symbol = num_cols[0]

        # 7 parametrów konfiguracyjnych agenta RL
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        # obserwacja: [buy_p, sell_p, price, balance]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.model_outputs = []
        self.precompute_model_outputs()
        self.reset()

    def precompute_model_outputs(self):
        """Wstępne obliczenie predykcji modelu ML dla całych danych"""
        history = self.data.copy()
        features = getattr(self.model, "feature_names_in_", None)
        if features is None:
            features = history.select_dtypes(include=[np.number]).columns.tolist()
        features = list(features)

        x = history[features].values
        probs = self.model.predict_proba(x)
        classes = [str(c) for c in getattr(self.model, "classes_", [])]
        if not classes:
            classes = [f"class_{i}" for i in range(probs.shape[1])]
        self.model_outputs = [dict(zip(classes, p.astype(float).tolist())) for p in probs]

        if "buy" not in classes or "sell" not in classes:
            print("[TradingEnv] Warning: model.classes_ doesn't contain 'buy'/'sell'. Found:", classes)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.wallet = Wallet(self.initial_balance, [self.symbol], WalletConfig())
        self.decision = DecisionEngine(DecisionConfig())
        self.total_reward = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        price = float(self.data.iloc[self.t][self.symbol])
        prob = self.model_outputs[self.t]
        buy_p = float(prob.get("buy", 0.0))
        sell_p = float(prob.get("sell", 0.0))
        obs = np.array([buy_p, sell_p, price, float(self.wallet.balance)], dtype=np.float32)
        return obs

    def step(self, action):
        # skalowanie [-1,1] -> [0,1]
        scaled_action = (action + 1.0) / 2.0

        # aktualizacja parametrów decyzyjnych
        self.decision.config.threshold_buy = scaled_action[0]
        self.decision.config.threshold_sell = scaled_action[1]
        self.decision.config.tx_fraction = scaled_action[2]
        self.decision.config.max_fraction_when_confident = scaled_action[3]
        self.decision.config.mincost = scaled_action[4] * 100
        self.wallet.config.stop_loss = scaled_action[5]
        self.wallet.config.max_wallet_percentage_crypto = scaled_action[6]

        price = float(self.data.iloc[self.t][self.symbol])
        prob_map = self.model_outputs[self.t]
        decision = self.decision.decide(prob_map, self.wallet, self.symbol, price)

        if decision is None or decision.get("qty") is None:
            decision = {"action": "hold", "qty": 0.0}

        qty = decision.get("qty", 0.0)
        action_type = decision.get("action", "hold")

        # uwzględnienie fee przy każdej transakcji
        if action_type == "buy" and qty > 0:
            cost = qty * price * (1 + self.fee)
            if cost >= self.decision.config.mincost:
                self.wallet.execute_order("buy", self.symbol, qty, price)
                self.wallet.balance -= qty * price * self.fee
        elif action_type == "sell" and qty > 0:
            proceeds = qty * price * (1 - self.fee)
            self.wallet.execute_order("sell", self.symbol, qty, price)
            self.wallet.balance -= qty * price * self.fee

        portfolio_value = self.wallet.total_value({self.symbol: price}) or 0.0
        reward = portfolio_value - self.total_reward
        self.total_reward = portfolio_value

        self.t += 1
        terminated = self.t >= len(self.data) - 1
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), terminated, truncated, {}
