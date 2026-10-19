from utils.explainers.shap_explainer import ShapExplainer
from utils.explainers.permutation_explainer import PermutationExplainer
from utils.explainers.lime_explainer import LimeExplainer

class XaiFactory:
    @staticmethod
    def create(method, pipeline, X_train=None, task='classification', config=None):
        config = config or {}
        common = dict(task=task, class_index=config.get('xai_class_index'),
                      random_state=config.get('xai_seed', 42))
        if method == 'shap':
            return ShapExplainer(pipeline, X_train, **common,
                                 background_size=config.get('shap_background_size', 50),
                                 permutations=config.get('shap_permutations', 3))
        if method == 'lime':
            return LimeExplainer(pipeline, X_train, **common,
                                 num_samples=config.get('lime_num_samples', 5000))
        if method == 'permutation':
            return PermutationExplainer(pipeline)
        raise ValueError(f'Unknown XAI method: {method}')
