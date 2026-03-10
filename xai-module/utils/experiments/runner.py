import numpy as np
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from utils.data.loaders.generic_loader import GenericDataLoader
from utils.preprocessing.pipeline_preprocessor import GenericPreprocessor
from utils.models.model_factory import ModelFactory
from utils.evaluation.metrics import (
    evaluate_classification,
    evaluate_regression
)


class ExperimentRunner:
    def __init__(self, config: dict, model_name: str):
        self.config = config
        self.model_name = model_name

    def run(self):
        loader = GenericDataLoader(self.config)
        df = loader.load()

        preprocessor = GenericPreprocessor(
            target=self.config["target"],
            split_type=self.config.get("split_type", "random"),
            test_size=self.config.get("test_size", 0.2),
            scaling=True
        )

        X_train, X_test, y_train, y_test, transformer = preprocessor.prepare(df)

        model = ModelFactory.create(
            model_name=self.model_name,
            task=self.config["task"]
        )

        if self.config["task"] == "classification" and (
            isinstance(model, XGBClassifier) or self.model_name == "xgb"
        ):
            classes = np.unique(np.asarray(y_train))
            n_classes = int(len(classes))

            if n_classes == 0:
                raise RuntimeError("XGBoost: n_classes == 0. y_train jest puste.")

            if n_classes == 1:
                raise RuntimeError(
                    f"XGBoost: w y_train jest tylko jedna klasa ({classes[0]})."
                )

            if n_classes == 2:
                model.set_params(
                    objective="binary:logistic",
                    eval_metric="logloss"
                )
                try:
                    model.set_params(num_class=None)
                except Exception:
                    pass
            else:
                model.set_params(
                    objective="multi:softprob",
                    num_class=n_classes,
                    eval_metric="mlogloss"
                )

        pipeline = Pipeline([
            ("preprocessing", transformer),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        if self.config["task"] == "classification":
            metrics = evaluate_classification(y_test, y_pred)
        else:
            metrics = evaluate_regression(y_test, y_pred)

        print("📊 Metrics:", metrics)

        return {
            "pipeline": pipeline,
            "metrics": metrics,
            "X_test": X_test,
            "y_test": y_test,
            "X_train": X_train,
            "y_train": y_train,
            "y_pred": y_pred
        }