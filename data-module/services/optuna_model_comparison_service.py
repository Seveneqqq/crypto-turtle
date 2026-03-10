import pandas as pd
from collections import Counter


class OptunaModelComparisonService:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _extract_features(self, row):

        if isinstance(row.get("features"), list):
            return row["features"]

        if isinstance(row.get("feature_combo"), list):
            return row["feature_combo"]

        return []

    def sklearn_feature_ranking(self):

        counter = Counter()

        df = self.df[self.df["accuracy"].notna()]

        for _, r in df.iterrows():

            features = self._extract_features(r)

            for f in features:
                counter[f] += 1

        result = pd.DataFrame(counter.items(), columns=["feature", "count"])

        return result.sort_values("count", ascending=False)

    def lstm_feature_ranking(self):

        counter = Counter()

        df = self.df[self.df["val_loss"].notna()]

        for _, r in df.iterrows():

            features = self._extract_features(r)

            for f in features:
                counter[f] += 1

        result = pd.DataFrame(counter.items(), columns=["feature", "count"])

        return result.sort_values("count", ascending=False)