import pandas as pd
from collections import Counter


class OptunaAnalyticsService:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _extract_features(self, row):

        if isinstance(row.get("features"), list):
            return row["features"]

        if isinstance(row.get("feature_combo"), list):
            return row["feature_combo"]

        return []

    def feature_importance(self):

        counter = Counter()

        for _, r in self.df.iterrows():

            features = self._extract_features(r)

            for f in features:
                counter[f] += 1

        result = pd.DataFrame(counter.items(), columns=["feature", "count"])

        return result.sort_values("count", ascending=False)

    def preprocessing_impact(self):

        if "accuracy" not in self.df.columns:
            return pd.Series()

        df = self.df[self.df["accuracy"].notna()]

        if df.empty:
            return pd.Series()

        return (
            df.groupby("preprocessing")["accuracy"]
            .mean()
            .sort_values(ascending=False)
        )

    def horizon_impact(self):

        if "accuracy" not in self.df.columns:
            return pd.Series()

        df = self.df[self.df["accuracy"].notna()]

        if df.empty:
            return pd.Series()

        return (
            df.groupby("horizon")["accuracy"]
            .mean()
        )