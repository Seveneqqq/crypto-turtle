import pandas as pd


class FeatureImportanceReport:

    @staticmethod
    def build_table(importance_dict, top_n=20):

        rows = []

        for i, (feature, value) in enumerate(importance_dict.items()):

            rows.append({
                "rank": i + 1,
                "feature": feature,
                "importance": value
            })

        df = pd.DataFrame(rows)

        return df.head(top_n)