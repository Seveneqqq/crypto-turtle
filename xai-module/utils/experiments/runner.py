import numpy as np
import hashlib
import json
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
            scaling=True,
            random_state=self.config.get("split_seed", 42),
            time_column=self.config.get("time_column"),

            # ===== POPRAWKA =====
            finance_horizon=self.config.get("finance_horizon", 1),
            finance_target_type=self.config.get(
                "finance_target_type",
                "return"
            )
            # ====================
        )

        X_train, X_test, y_train, y_test, transformer = preprocessor.prepare(df)

        model = ModelFactory.create(
            model_name=self.model_name,
            task=self.config["task"]
        )

        if "random_state" in model.get_params():
            model.set_params(random_state=self.config.get("model_seed", 42))
        if "n_jobs" in model.get_params() and "n_jobs" in self.config:
            model.set_params(n_jobs=self.config["n_jobs"])
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

        print(" Metrics:", metrics)

        return {
            "data_preparation": {
                "chronology": preprocessor.chronology_,
                "split_type": self.config.get("split_type", "random"),
                "split_seed": self.config.get("split_seed", 42),
                "model_seed": self.config.get("model_seed", 42),
                "train_rows":len(X_train), "test_rows":len(X_test),
                "split_metadata":getattr(preprocessor, "split_metadata_", {}),
                "train_indices_sha256":hashlib.sha256(json.dumps(X_train.index.tolist(),default=str).encode()).hexdigest(),
                "test_indices_sha256":hashlib.sha256(json.dumps(X_test.index.tolist(),default=str).encode()).hexdigest(),
                "encoded_class_labels":preprocessor.label_encoder.classes_.tolist() if preprocessor.label_encoder is not None else None,
                "fitted_model_parameters":model.get_params()},
            "pipeline": pipeline,
            "metrics": metrics,
            "X_test": X_test,
            "y_test": y_test,
            "X_train": X_train,
            "y_train": y_train,
            "y_pred": y_pred
        }