import gymnasium as gym
from gymnasium import spaces
import numpy as np
from services.rl.wallet import Wallet, WalletConfig
from services.rl.decision import DecisionEngine, DecisionConfig

class TradingEnv(gym.Env):
    def __init__(self, data, model, initial_balance=1000.0, fee=0.001):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.model = model
        self.fee = float(fee)
        self.initial_balance = float(initial_balance)

        # wybór symbolu ceny
        if "close" in data.columns:
            self.symbol = "close"
        else:
            num_cols = data.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                raise ValueError("Brak kolumn numerycznych w danych.")
            self.symbol = num_cols[0]

        # akcje: threshold_buy, threshold_sell, tx_fraction
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # obserwacja znormalizowana: [buy_p, sell_p, price_norm, balance_norm]
        # price_norm = price / initial_price, balance_norm = balance / initial_balance
        # ograniczone do rozsądnego zakresu, żeby uniknąć -inf/inf w sieci
        low = np.array([0.0, 0.0, 0.0, 0.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0], dtype=np.float32)
        high = np.array([1.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)  # price_norm, balance_norm rzadko >10
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # precompute ML outputs
        self.model_outputs = []
        self.precompute_model_outputs()

        # initial price reference for normalization
        self.initial_price = float(self.data.iloc[0][self.symbol])

        self.reset()

    def precompute_model_outputs(self):
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
        # prev portfolio value used to compute relative reward
        self.prev_value = self.wallet.total_value({self.symbol: float(self.data.iloc[self.t][self.symbol])}) or self.initial_balance
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.data.iloc[self.t]
        price = float(self.data.iloc[self.t][self.symbol])
        prob = self.model_outputs[self.t]
        buy_p = float(prob.get("buy", 0.0))
        sell_p = float(prob.get("sell", 0.0))

        ret1 = float(row.get("ret1", 0.0))
        ret1_lag1 = float(row.get("ret1_lag1", 0.0))
        roll_mean_10 = float(row.get("roll_mean_10", 0.0))
        roll_std_10 = float(row.get("roll_std_10", 0.0))
        macd = float(row.get("macd", 0.0))
        macd_sig = float(row.get("macd_sig", 0.0))
        atr = float(row.get("ATR_14", 0.0))
        vol = float(row.get("volume", 0.0))

        # normalizacja względem początkowej ceny i początkowego salda
        price_norm = price / (self.initial_price + 1e-12)
        balance_norm = (float(self.wallet.balance) / (self.initial_balance + 1e-12))

        obs = np.array([
            buy_p, sell_p, price_norm, balance_norm,
            ret1, ret1_lag1, roll_mean_10, roll_std_10, macd, macd_sig, atr, vol
        ], dtype=np.float32)

        # zabezpieczenia numeryczne
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        obs = np.round(obs, 4)

        # upewnij się, że mieści się w zadanym zakresie obserwacji
        # (clip zgodnie z observation_space)
        low = self.observation_space.low
        high = self.observation_space.high
        obs = np.clip(obs, low, high).astype(np.float32)

        return obs

    def step(self, action):
        # sanity: action shape
        action = np.asarray(action, dtype=np.float32).flatten()
        if action.size != 3:
            raise ValueError("Expected action shape (3,), got %s" % (action.shape,))

        # skalowanie [-1,1] -> [0,1]
        scaled = (action + 1.0) / 2.0
        # mapowanie thresholds do sensownego podzakresu (blisko 0.5, aby uniknąć ekstremów)
        # thresholds w zakresie [0.2, 0.8] (ułatwia eksplorację wokół dobrego miejsca)
        thr_low, thr_high = 0.2, 0.8
        threshold_buy = thr_low + scaled[0] * (thr_high - thr_low)
        threshold_sell = thr_low + scaled[1] * (thr_high - thr_low)

        # tx_fraction w zakresie [0.001, 0.2] (0.1% - 20% kapitału)
        tx_min, tx_max = 0.001, 0.2
        tx_fraction = tx_min + scaled[2] * (tx_max - tx_min)

        # ustaw parametry decyzyjne
        self.decision.config.threshold_buy = float(threshold_buy)
        self.decision.config.threshold_sell = float(threshold_sell)
        self.decision.config.tx_fraction = float(tx_fraction)

        # pozostałe parametry stałe (można potem wrócić)
        self.decision.config.max_fraction_when_confident = 0.5
        self.decision.config.mincost = 1.0
        self.wallet.config.stop_loss = 1.0
        self.wallet.config.max_wallet_percentage_crypto = 1.0

        # wykonaj decyzję na aktualnym kroku
        price = float(self.data.iloc[self.t][self.symbol])
        prob_map = self.model_outputs[self.t]
        decision = self.decision.decide(prob_map, self.wallet, self.symbol, price)

        if decision is None or decision.get("qty") is None:
            decision = {"action": "hold", "qty": 0.0}

        qty = float(decision.get("qty", 0.0))
        action_type = decision.get("action", "hold")

        # uwzględnienie fee przy każdej transakcji (i mincost)
        if action_type == "buy" and qty > 0:
            cost = qty * price * (1 + self.fee)
            if cost >= self.decision.config.mincost:
                # execute_order powinien wewnętrznie aktualizować holdings i balance
                self.wallet.execute_order("buy", self.symbol, qty, price)
                # doprecyzowanie: odciągamy fee w gotówce (jeżeli execute_order nie bije fee)
                self.wallet.balance -= qty * price * self.fee
        elif action_type == "sell" and qty > 0:
            proceeds = qty * price * (1 - self.fee)
            self.wallet.execute_order("sell", self.symbol, qty, price)
            self.wallet.balance -= qty * price * self.fee

        # portfolio value i względny reward (normalizowany)
        portfolio_value = self.wallet.total_value({self.symbol: price}) or 0.0
        # względna zmiana wartości względem poprzedniego kroku
        prev_value = float(self.prev_value) if hasattr(self, "prev_value") else float(self.initial_balance)
        reward = 0.0
        if prev_value > 0:
            reward = (portfolio_value - prev_value) / prev_value
        # clip reward do rozsądnego zakresu, aby nie dać eksplodujących gradientów
        reward = float(np.clip(reward, -0.2, 0.2))

        # update prev_value
        self.prev_value = portfolio_value

        # postęp czasu
        self.t += 1
        terminated = self.t >= len(self.data) - 1
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "portfolio_value": float(portfolio_value),
            "threshold_buy": self.decision.config.threshold_buy,
            "threshold_sell": self.decision.config.threshold_sell,
            "tx_fraction": self.decision.config.tx_fraction,
        }

        return obs, reward, terminated, truncated, info
