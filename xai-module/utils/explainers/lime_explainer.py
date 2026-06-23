import re
import lime.lime_tabular
import numpy as np


class LimeExplainer:
    def __init__(self, pipeline, X_train, task="classification"):
        self.pipeline = pipeline
        self.model = pipeline.named_steps["model"]
        self.preprocessor = pipeline.named_steps["preprocessing"]
        self.task = task

        X_transformed = self.preprocessor.transform(X_train)

        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        self.feature_names = self.preprocessor.get_feature_names_out()

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_transformed,
            feature_names=self.feature_names,
            mode=self.task,
            discretize_continuous=True
        )

    def _normalize_feature_name(self, name: str) -> str:
        for feature in self.feature_names:
            if feature in name:
                return feature
        return name

    def explain(self, X_instance):
        X_transformed = self.preprocessor.transform(
            X_instance.to_frame().T
        )

        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        if self.task == "classification":
            explanation = self.explainer.explain_instance(
                X_transformed[0],
                self.model.predict_proba
            )
        else:
            explanation = self.explainer.explain_instance(
                X_transformed[0],
                self.model.predict
            )

        normalized = {}

        for feature, value in explanation.as_list():
            clean_feature = self._normalize_feature_name(feature)
            normalized[clean_feature] = normalized.get(clean_feature, 0) + abs(float(value))

        return dict(
            sorted(
                normalized.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )