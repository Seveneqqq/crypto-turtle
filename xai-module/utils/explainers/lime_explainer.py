import lime.lime_tabular
import numpy as np


class LimeExplainer:

    def __init__(self, pipeline, X_train):

        self.pipeline = pipeline
        self.model = pipeline.named_steps["model"]
        self.preprocessor = pipeline.named_steps["preprocessing"]

        X_transformed = self.preprocessor.transform(X_train)

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_transformed,
            mode="classification",
            discretize_continuous=True
        )

    def explain_instance(self, X_instance):

        X_transformed = self.preprocessor.transform(X_instance)

        explanation = self.explainer.explain_instance(
            X_transformed[0],
            self.model.predict_proba
        )

        return explanation