from utils.explainers.shap_explainer import ShapExplainer
from utils.explainers.permutation_explainer import PermutationExplainer
from utils.explainers.lime_explainer import LimeExplainer


class XaiFactory:
    @staticmethod
    def create(method: str, pipeline, X_train=None, task="classification"):
        if method == "shap":
            return ShapExplainer(pipeline)

        if method == "permutation":
            return PermutationExplainer(pipeline)

        if method == "lime":
            return LimeExplainer(pipeline, X_train, task=task)

        raise ValueError(f"Unknown XAI method: {method}")