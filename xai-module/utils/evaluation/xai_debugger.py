import numpy as np
import pandas as pd


class XaiDebugger:
    @staticmethod
    def find_errors(task, y_true, y_pred):
        if task == "classification":
            mask = y_true != y_pred
        else:
            errors = np.abs(np.array(y_true) - np.array(y_pred))
            threshold = np.quantile(errors, 0.90)
            mask = errors >= threshold

        return mask

    @staticmethod
    def explain_errors(task, method, explainer, X_test, y_true, y_pred, max_cases=5):
        mask = XaiDebugger.find_errors(task, y_true, y_pred)

        if isinstance(mask, pd.Series):
            X_errors = X_test.loc[mask]
        else:
            X_errors = X_test.iloc[np.where(mask)[0]]

        X_errors = X_errors.iloc[:max_cases]

        debug_results = []

        if len(X_errors) == 0:
            return debug_results

        if method == "shap":
            shap_values = explainer.explain(X_errors)
            feature_names = explainer.preprocessor.get_feature_names_out()

            for i in range(len(X_errors)):
                pairs = sorted(
                    zip(feature_names, shap_values[i]),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )

                idx = X_errors.index[i]
                debug_results.append({
                    "sample_index": int(idx),
                    "top_features": [
                        {"feature": f, "value": float(v)}
                        for f, v in pairs[:10]
                    ]
                })

        elif method == "lime":
            for i in range(len(X_errors)):
                idx = X_errors.index[i]
                explanation = explainer.explain(X_errors.iloc[i])
                pairs = sorted(
                    explanation.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )

                debug_results.append({
                    "sample_index": int(idx),
                    "top_features": [
                        {"feature": f, "value": float(v)}
                        for f, v in pairs[:10]
                    ]
                })

        elif method == "permutation":
            debug_results.append({
                "sample_index": None,
                "top_features": [],
                "note": "Permutation importance is global, not suited for single error case explanations."
            })

        return debug_results