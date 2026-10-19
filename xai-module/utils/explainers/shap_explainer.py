import numpy as np
import shap
from utils.explainers.output_contract import OutputContract

class ShapExplainer(OutputContract):
    def __init__(self, pipeline, X_train, task='classification', class_index=None,
                 random_state=42, background_size=50, permutations=3):
        super().__init__(pipeline, task, class_index)
        if background_size < 1 or permutations < 1:
            raise ValueError('SHAP background size and permutation count must be positive')
        self.pipeline = pipeline
        self.random_state = random_state
        self.permutations = permutations
        background = X_train.sample(min(background_size, len(X_train)), random_state=random_state)
        self.background_indices = background.index.tolist()
        self.background = self.transform(background)
        self.feature_names = self.preprocessor.get_feature_names_out()

    def explain(self, X_sample, random_state=None):
        seed = self.random_state if random_state is None else random_state
        masker = shap.maskers.Independent(self.background, max_samples=len(self.background))
        explainer = shap.PermutationExplainer(self.predict_output, masker,
                                            feature_names=list(self.feature_names), seed=seed)
        result = explainer(self.transform(X_sample),
                           max_evals=(2*len(self.feature_names)+1)*self.permutations, silent=True)
        self.last_base_values = np.asarray(result.base_values)
        return np.asarray(result.values)

    def get_global_importance(self, shap_values):
        values = np.abs(np.atleast_2d(shap_values)).mean(axis=0)
        return dict(sorted(zip(self.feature_names, map(float, values)),
                           key=lambda pair: pair[1], reverse=True))
