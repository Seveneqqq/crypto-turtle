import numpy as np
from sklearn.inspection import permutation_importance


class PermutationExplainer:

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def output_metadata(self):
        return {"output": "estimator_default_score", "class_index": None,
                "class_label": None, "scope": "global_permutation_importance"}

    def explain(self, X_test, y_test, random_state=42):
        if not X_test.columns.is_unique:
            raise ValueError("PFI requires unique input column names")
        result = permutation_importance(
            estimator=self.pipeline,
            X=X_test,
            y=y_test,
            n_repeats=3,
            random_state=random_state,
            n_jobs=-1
        )

        result["feature_names"] = list(X_test.columns)
        return result

    def get_global_importance(self, perm_result):
        if "feature_names" not in perm_result:
            raise ValueError("PFI result lacks input column metadata; recompute PFI")
        feature_names = perm_result["feature_names"]
        importances = np.asarray(perm_result.importances_mean)
        if len(importances) != len(feature_names):
            raise ValueError("PFI values do not match recorded input columns")

        sorted_idx = np.argsort(importances)[::-1]

        return {
            feature_names[i]: float(importances[i])
            for i in sorted_idx
        }
