import gymnasium as gym
from gymnasium import spaces
import numpy as np
from services.rl.wallet import Wallet, WalletConfig
from services.rl.decision import DecisionEngine, DecisionConfig

class TradingEnv(gym.Env):
    def __init__(self, data, model, initial_balance=1000.0):
        super().__init__()
        self.data = data
        self.model = model
        self.symbol = data.columns[0]
        self.initial_balance = initial_balance

        # akcja = 7 parametrów uczonych przez agenta
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        # obserwacja: sygnał + cena + holdings + balance
        # Uwaga: price/balance mogą być >1, lepiej ustawić bounds na duże wartości:
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # przelicz wyniki modelu raz
        self.model_outputs = []
        self.precompute_model_outputs()

        self.reset()

    def precompute_model_outputs(self):
        """Oblicz wyniki modelu dla wszystkich wierszy danych na starcie"""
        history = self.data.copy()

        # próbujemy pobrać cechy z modelu; jeśli nie ma, bierzemy wszystkie kolumny poza symbolową
        features = getattr(self.model, "feature_names_in_", None)
        if features is None:
            # załóżmy że kolumna symbolu to np. 'close' lub nazwa pierwszej kolumny
            # wybieramy kolumny numeryczne
            features = history.select_dtypes(include=[np.number]).columns.tolist()
            # jeśli symbol (np. kolumna z ceną) też jest w features nie usuwamy; model wybierze odpowiednie
        # upewnij się, że features są listą (np. np.ndarray -> list)
        features = list(features)

        # przygotuj dane wejściowe i policz predykcje
        x = history[features].values
        probs = self.model.predict_proba(x)  # shape (n_rows, n_classes)

        # etykiety klas (np. ['buy','hold','sell']) -> konwertujemy na string dla spójności
        classes = [str(c) for c in getattr(self.model, "classes_", [])]

        # zabezpieczenie: jeśli brak klas, zbuduj domyślne nazwy
        if not classes:
            classes = [f"class_{i}" for i in range(probs.shape[1])]

        # Zamieniamy każdy wiersz prawdopodobieństw na dict {label: prob}
        self.model_outputs = [dict(zip(classes, p.astype(float).tolist())) for p in probs]

            # Debug: poinformuj, jeśli etykiety 'buy' lub 'sell' nie występują
        if "buy" not in classes or "sell" not in classes:
            print("[TradingEnv.precompute_model_outputs] Warning: model.classes_ doesn't contain 'buy' and/or 'sell'. Classes:", classes)

        print("precomputing model outputs done, rows =", len(self.model_outputs))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.wallet = Wallet(self.initial_balance, [self.symbol], WalletConfig())
        self.decision = DecisionEngine(DecisionConfig())
        self.total_reward = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        price = float(self.data.iloc[self.t][self.symbol])
        prob = self.model_outputs[self.t]  # dict: {label: prob}
        # bezpiecznie pobieramy prawdopodobieństwa (0 jeśli brak)
        buy_p = float(prob.get("buy", 0.0))
        sell_p = float(prob.get("sell", 0.0))
        obs = np.array([buy_p, sell_p, price, float(self.wallet.balance)], dtype=np.float32)
        return obs

    def step(self, action):
        # ustaw parametry od agenta
        scaled_action = (action + 1.0) / 2.0  # mapowanie z [-1,1] -> [0,1]

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
            decision = {"action": "hold", "qty": 0.0, "model_value": 0.0, "reason": "invalid decision"}

        self.wallet.execute_order(decision['action'], self.symbol, decision['qty'], price)

        # nagroda = zmiana wartości portfela
        portfolio_value = self.wallet.total_value({self.symbol: price}) or 0.0
        reward = portfolio_value - self.total_reward
        self.total_reward = portfolio_value

        self.t += 1
        terminated = self.t >= len(self.data) - 1
        truncated = False  # albo warunek np. max_steps

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, float(reward), terminated, truncated, {}

