from sklearn.base import BaseEstimator, TransformerMixin
import ta
import numpy as np
import pandas as pd

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # EMA
        for w in [10, 20]:
            df[f"ema_{w}"] = ta.trend.EMAIndicator(df["close"], window=w).ema_indicator()

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

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()

        # Standard deviation
        df["std_20"] = df["close"].rolling(window=20).std()

        # OBV
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()

        # 🔹 Zamiast cumvol → rolling wolumen
        df["vol_mean_20"] = df["volume"].rolling(window=20).mean()
        df["vol_std_20"] = df["volume"].rolling(window=20).std()

        # VWAP
        vwap = ta.volume.VolumeWeightedAveragePrice(
            df["high"], df["low"], df["close"], df["volume"], window=14
        )
        df["vwap"] = vwap.volume_weighted_average_price()

        # 🔹 Momentum i log returns
        df["log_return"] = np.log(df["close"]).diff()
        df["momentum_3"] = df["close"] / df["close"].shift(3) - 1
        df["momentum_7"] = df["close"] / df["close"].shift(7) - 1
        df["log_return_mean_5"] = df["log_return"].rolling(window=5).mean()
        df["log_return_std_5"] = df["log_return"].rolling(window=5).std()

        # 🔹 Wypełnianie braków + stabilność wskaźników
        df = df.bfill().ffill()

        return df
