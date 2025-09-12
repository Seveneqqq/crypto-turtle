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

        # ATR
        atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14)
        df["atr"] = atr.average_true_range()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()

        # Standard deviation
        df["std_20"] = df["close"].rolling(window=20).std()

        # Cumulative volume
        df["cumvol"] = df["volume"].cumsum()

        # VWAP
        vwap = ta.volume.VolumeWeightedAveragePrice(
            df["high"], df["low"], df["close"], df["volume"], window=14
        )
        df["vwap"] = vwap.volume_weighted_average_price()

        # 🔹 Normalizacja wolumenu
        df["volume_norm"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # 🔹 Momentum
        df["momentum_10"] = df["close"].diff(10)

        # 🔹 Log returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        return df.bfill().ffill()
