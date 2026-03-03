from sklearn.inspection import permutation_importance


class PermutationExplainer:

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.model = pipeline.named_steps["model"]
        self.preprocessor = pipeline.named_steps["preprocessing"]

    def explain(self, X_test, y_test):

        X_transformed = self.preprocessor.transform(X_test)

        result = permutation_importance(
            self.model,
            X_transformed,
            y_test,
            n_repeats=3,
            random_state=42,
            n_jobs=-1
        )

        return result

    def get_global_importance(self, perm_result):

        feature_names = self.preprocessor.get_feature_names_out()

        sorted_idx = perm_result.importances_mean.argsort()[::-1]

        return {
            feature_names[i]: float(perm_result.importances_mean[i])
            for i in sorted_idx
        }