import time
import numpy as np

from utils.experiments.runner import ExperimentRunner
from utils.explainers.shap_explainer import ShapExplainer
from utils.explainers.permutation_explainer import PermutationExplainer


class ExperimentPipeline:

    def __init__(self, config: dict, model_names: list):
        self.config = config
        self.model_names = model_names

    def run(self):

        results = {}

        for model_name in self.model_names:

            print("\n============================")
            print(f"🚀 Running model: {model_name}")
            print("============================")

            start_model = time.time()

            runner = ExperimentRunner(self.config, model_name)
            result = runner.run()

            pipeline = result["pipeline"]
            X_train = result["X_train"]
            X_test = result["X_test"]
            y_test = result["y_test"]

            # ======================
            # SHAP
            # ======================
            print("🔥 Running SHAP...")
            start_shap = time.time()

            shap_explainer = ShapExplainer(pipeline)
            sample = X_train.sample(50, random_state=42)

            shap_values = shap_explainer.explain(sample)
            shap_importance = shap_explainer.get_global_importance(
                shap_values
            )

            print(f"⏱ SHAP time: {time.time() - start_shap:.2f}s")

            print("📊 SHAP top 5 features:")
            for i, (k, v) in enumerate(shap_importance.items()):
                if i == 5:
                    break
                print(f"   {k}: {v:.6f}")

            # ======================
            # Permutation
            # ======================
            print("🔥 Running Permutation...")
            start_perm = time.time()

            perm_explainer = PermutationExplainer(pipeline)

            perm_result = perm_explainer.explain(
                X_test.iloc[:300],
                y_test.iloc[:300]
            )

            perm_importance = perm_explainer.get_global_importance(
                perm_result
            )

            print(f"⏱ Permutation time: {time.time() - start_perm:.2f}s")

            print("📊 Permutation top 5 features:")
            for i, (k, v) in enumerate(perm_importance.items()):
                if i == 5:
                    break
                print(f"   {k}: {v:.6f}")

            print(f"⏱ Model total time: {time.time() - start_model:.2f}s")

            results[model_name] = {
                "metrics": result["metrics"],
                "shap_importance": shap_importance,
                "permutation_importance": perm_importance
            }

        return results