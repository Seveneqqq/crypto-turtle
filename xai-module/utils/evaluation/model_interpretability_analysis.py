import pandas as pd

class ModelInterpretabilityAnalysis:

    @staticmethod
    def build_table(model_results):

        rows = []

        for model_name, result in model_results.items():

            accuracy = result["metrics"].get("accuracy") or result["metrics"].get("f1_macro")

            interpretability = 0

            for method in result["xai"]:
                interpretability += result["xai"][method]["stability"]

            interpretability /= len(result["xai"])

            rows.append({
                "model": model_name,
                "accuracy": accuracy,
                "interpretability": interpretability
            })

        return pd.DataFrame(rows)