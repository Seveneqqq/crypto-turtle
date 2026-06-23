import numpy as np
from sklearn.inspection import permutation_importance


class PermutationExplainer:

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def explain(self, X_test, y_test):
        result = permutation_importance(
            estimator=self.pipeline,
            X=X_test,
            y=y_test,
            n_repeats=3,
            random_state=42,
            n_jobs=-1
        )

        return result

    def get_global_importance(self, perm_result):
        preprocessor = self.pipeline.named_steps["preprocessing"]
        feature_names = preprocessor.get_feature_names_out()

        importances = np.asarray(perm_result.importances_mean)

        if len(importances) != len(feature_names):
            feature_names = [
                f"feature_{i}"
                for i in range(len(importances))
            ]

        sorted_idx = np.argsort(importances)[::-1]

        return {
            feature_names[i]: float(importances[i])
            for i in sorted_idx
        }