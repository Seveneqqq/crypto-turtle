class XaiLocal:
    @staticmethod
    def generate_local_explanations(method, explainer, X_sample, max_samples=5):
        results = []

        X_local = X_sample.iloc[:max_samples]

        if method == "shap":
            shap_values = explainer.explain(X_local)
            feature_names = explainer.preprocessor.get_feature_names_out()

            for i in range(len(X_local)):
                row_vals = shap_values[i]
                pairs = sorted(
                    zip(feature_names, row_vals),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )

                results.append({
                    "sample_index": int(X_local.index[i]),
                    "top_features": [
                        {"feature": f, "value": float(v)}
                        for f, v in pairs[:10]
                    ]
                })

        elif method == "lime":
            for i in range(len(X_local)):
                explanation = explainer.explain(X_local.iloc[i])
                pairs = sorted(
                    explanation.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )

                results.append({
                    "sample_index": int(X_local.index[i]),
                    "top_features": [
                        {"feature": f, "value": float(v)}
                        for f, v in pairs[:10]
                    ]
                })

        elif method == "permutation":
            results.append({
                "sample_index": None,
                "top_features": [],
                "note": "Permutation importance is global, not local."
            })

        return results