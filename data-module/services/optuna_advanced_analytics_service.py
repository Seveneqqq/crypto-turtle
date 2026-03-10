import pandas as pd
from collections import Counter


class OptunaAdvancedAnalyticsService:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _extract_features(self, row):
        if isinstance(row.get("features"), list):
            return row["features"]

        if isinstance(row.get("feature_combo"), list):
            return row["feature_combo"]

        return []

    def build_feature_table(self):
        
        rows = []
    
        for _, r in self.df.iterrows():
        
            if r.get("score") is None:
                continue
            
            features = self._extract_features(r)
    
            for f in features:
            
                rows.append({
                    "feature": f,
                    "score": r["score"]
                })
    
        return pd.DataFrame(rows)

    def feature_importance(self):

        ft = self.build_feature_table()

        return (
            ft.groupby("feature")["score"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

    def preprocessing_ranking(self):

        df = self.df.copy()

        df = df[df["accuracy"].notna()]

        return (
            df.groupby("preprocessing")["accuracy"]
            .mean()
            .sort_values(ascending=False)
        )

    def horizon_ranking(self):

        df = self.df.copy()

        df = df[df["accuracy"].notna()]

        return (
            df.groupby("horizon")["accuracy"]
            .mean()
            .sort_index()
        )

    def dataset_ranking(self):

        df = self.df.copy()

        df = df[df["accuracy"].notna()]

        return (
            df.groupby("filename")["accuracy"]
            .mean()
            .sort_values(ascending=False)
        )

    def top_feature_combos(self, top_n=10):

        rows = []

        for _, r in self.df.iterrows():

            if isinstance(r.get("features"), list):
                rows.append(
                    {
                        "features": ",".join(r["features"]),
                        "score": r["accuracy"]
                    }
                )

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        return df.sort_values("score", ascending=False).head(top_n)