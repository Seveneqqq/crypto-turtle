import shap
import numpy as np


class ShapExplainer:

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.model = pipeline.named_steps["model"]
        self.preprocessor = pipeline.named_steps["preprocessing"]

    def explain(self, X_sample):

        X_transformed = self.preprocessor.transform(X_sample)

        # 1) Jeśli sparse -> dla SHAP na próbce zamień na dense (50 wierszy to OK)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        # 2) Upewnij się że float
        X_transformed = np.asarray(X_transformed, dtype=np.float64)

        # =========================
        # TREE MODELS (RF, XGB itp.)
        # =========================
        if hasattr(self.model, "feature_importances_"):
            explainer = shap.TreeExplainer(
                self.model,
                feature_perturbation="tree_path_dependent",
                model_output="raw"
            )

            shap_values = explainer.shap_values(
                X_transformed,
                check_additivity=False
            )

            # binary classification -> lista [class0, class1]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # multiclass -> (n, features, classes)
            elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
                # uśrednij po klasach, żeby mieć 2D (n, features)
                shap_values = np.mean(np.abs(shap_values), axis=2) * np.sign(np.mean(shap_values, axis=2))

            return shap_values

        # =========================
        # LINEAR MODELS (LogReg)
        # =========================
        if hasattr(self.model, "coef_"):
            # Background: mała próbka z X_transformed (wystarczy)
            background = X_transformed[:min(100, X_transformed.shape[0])]

            explainer = shap.LinearExplainer(
                self.model,
                background
            )

            shap_values = explainer.shap_values(X_transformed)

            # binary -> (n, features) OK
            # multiclass -> lista klas lub (n, features, classes) zależnie od wersji SHAP
            if isinstance(shap_values, list):
                # uśrednij po klasach -> 2D
                shap_values = np.mean(np.abs(np.stack(shap_values, axis=2)), axis=2)

            elif len(shap_values.shape) == 3:
                shap_values = np.mean(np.abs(shap_values), axis=2)

            return shap_values

        # =========================
        # Fallback (rzadko)
        # =========================
        # Jeśli tu trafisz, to lepiej użyć KernelExplainer/permutation,
        # ale to jest wolne i problematyczne przy wielu cechach.
        raise RuntimeError("Unsupported model type for SHAP in this implementation.")

    def get_global_importance(self, shap_values):

        feature_names = self.preprocessor.get_feature_names_out()

        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)

        mean_importance = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_importance)[::-1]

        return {
            feature_names[i]: float(mean_importance[i])
            for i in sorted_idx
        }