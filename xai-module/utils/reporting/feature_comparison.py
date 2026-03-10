import pandas as pd


class FeatureComparison:

    @staticmethod
    def build(method_importances):

        features = set()

        for imp in method_importances.values():
            features.update(imp.keys())

        rows = []

        for f in features:

            row = {"feature": f}

            for method, imp in method_importances.items():
                row[method] = imp.get(f, 0)

            rows.append(row)

        df = pd.DataFrame(rows)

        return df.sort_values(list(method_importances.keys())[0], ascending=False)