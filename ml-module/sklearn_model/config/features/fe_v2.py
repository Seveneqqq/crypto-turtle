from sklearn.base import BaseEstimator, TransformerMixin
import ta
import numpy as np

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # MA
        for w in [5, 10, 20]:
            df[f"ma_{w}"] = df["close"].rolling(window=w).mean()

        # SMA
        for w in [10, 30]:
            df[f"sma_{w}"] = ta.trend.SMAIndicator(df["close"], window=w).sma_indicator()

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # MACD
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()

        # ADX
        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
        df["adx"] = adx.adx()

        # ATR
        atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14)
        df["atr"] = atr.average_true_range()

        # Standard deviation
        df["std_20"] = df["close"].rolling(window=20).std()

        # Cumulative volume
        df["cumvol"] = df["volume"].cumsum()

        # 🔹 Normalizacja wolumenu (np. wolumen / rolling mean 20)
        df["volume_norm"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # 🔹 Momentum (prosta zmiana ceny)
        df["momentum_10"] = df["close"].diff(10)

        # 🔹 Log returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        return df.bfill().ffill()
