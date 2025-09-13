from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["return"] = df["close"].pct_change()
        df["ma_5"] = df["close"].rolling(window=5).mean()
        df["ma_15"] = df["close"].rolling(window=15).mean()
        df["ma_150"] = df["close"].rolling(window=150).mean()
        return df[["return", "ma_5", "ma_15", "ma_150"]]
