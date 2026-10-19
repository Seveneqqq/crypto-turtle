import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from utils.explainers.output_contract import OutputContract

class LimeExplainer(OutputContract):
    def __init__(self, pipeline, X_train, task='classification', class_index=None,
                 random_state=42, num_samples=5000):
        super().__init__(pipeline, task, class_index)
        self.pipeline = pipeline
        self.random_state = random_state
        self.num_samples = num_samples
        self.feature_names = self.preprocessor.get_feature_names_out()
        values = self.transform(X_train)
        categorical = [i for i in range(values.shape[1])
                       if set(np.unique(values[:, i])).issubset({0., 1.})]
        self.explainer = LimeTabularExplainer(
            values, feature_names=self.feature_names, mode=task,
            categorical_features=categorical, discretize_continuous=True,
            random_state=random_state)

    def explain(self, X_instance, random_state=None):
        seed = self.random_state if random_state is None else random_state
        rng = np.random.RandomState(seed)
        self.explainer.random_state = rng
        self.explainer.base.random_state = rng
        if self.explainer.discretizer is not None:
            self.explainer.discretizer.random_state = rng
        row = self.transform(X_instance.to_frame().T)[0]
        kwargs = {'num_features': len(self.feature_names), 'num_samples': self.num_samples}
        if self.task == 'classification':
            explanation = self.explainer.explain_instance(
                row, self.model.predict_proba, labels=(self.class_index,), **kwargs)
            label = self.class_index
        else:
            explanation = self.explainer.explain_instance(row, self.model.predict, **kwargs)
            label = 1  
        self.last_fit_score = float(explanation.score)
        weights = {str(self.feature_names[i]): float(weight)
                   for i, weight in explanation.as_map()[label]}
        return dict(sorted(weights.items(), key=lambda pair: abs(pair[1]), reverse=True))

    def representation_metadata(self):
        return {'weights': 'signed_local_surrogate_coefficients',
                'representation': 'discretized_interval_or_category_membership',
                'note': 'Coefficient magnitudes are not directly comparable with SHAP attribution units.'}
