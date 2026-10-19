import numpy as np
import pandas as pd
import os
import json
import traceback
from sklearn.pipeline import Pipeline
from PyALE import ale


class AleExplainer:
    """
    ALE Explainer
    - sklearn / Pipeline
    - feature może być None
    - zgodny kontrakt z SHAP / LIME
    """

    def __init__(self, logger=None):
        self.logger = logger
        if self.logger:
            self.logger.info("AleExplainer initialized")

    # ======================================================
    # MAIN
    # ======================================================
    def explain(
        self,
        model,
        data_path: str = None,
        X=None,
        instance: int = 0,
        feature: str = None,
        **kwargs
    ):
        if self.logger:
            self.logger.info("Starting ALE explanation")

        if X is not None:
            raise ValueError("ALE does not support LSTM models")

        if data_path is None:
            raise ValueError("data_path is required for ALE")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        # ================================
        # feature names jak SHAP
        # ================================
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
            df = df[feature_names]
        else:
            feature_names = list(df.columns)

        # ================================
        # WRAPPER wymagany przez PyALE
        # ================================
        class ModelWrapper:
            def __init__(self, model):
                self.model = model

            def predict(self, X):
                if hasattr(self.model, "predict_proba"):
                    preds = self.model.predict_proba(X)[:, 1]
                    return preds.astype(float)

                preds = self.model.predict(X)
                return np.array(preds, dtype=float)

        wrapped_model = ModelWrapper(model)

        # ================================
        # wybór feature
        # ================================
        if feature is None:
            features_to_compute = feature_names
        else:
            if feature not in feature_names:
                raise ValueError(f"Feature '{feature}' not found")
            features_to_compute = [feature]

        if self.logger:
            self.logger.info(f"Computing ALE for features: {features_to_compute}")

        shap_values = []
        sample_grid = []

        # ================================
        # ALE computation
        # ================================
        try:
            for feat in features_to_compute:
                ale_eff = ale(
                    X=df,
                    model=wrapped_model,  
                    feature=[feat],
                    include_CI=False,
                    plot=False
                )

                values = ale_eff["eff"].values.tolist()
                grid = ale_eff.index.values.tolist()

                shap_values.append(values)
                sample_grid.append(grid)

        except Exception:
            if self.logger:
                self.logger.exception("ALE computation failed")
            traceback.print_exc()
            raise

        expected_value = float(np.mean(wrapped_model.predict(df)))

        result = {
            "instance": instance,
            "model_type": "sklearn",
            "shap_values": shap_values,
            "expected_value": expected_value,
            "sample_data": sample_grid,
            "features": features_to_compute
        }

        self._save_result_json(result, explainer_name="ale")

        if self.logger:
            self.logger.info("ALE explanation finished successfully")

        return result

    # ======================================================
    # SAVE
    # ======================================================
    def _get_output_dir(self, model_type: str):
        if not self.logger:
            return None

        for handler in self.logger.handlers:
            if hasattr(handler, "baseFilename"):
                base_log_dir = os.path.dirname(handler.baseFilename)
                out_dir = os.path.join(base_log_dir, model_type)
                os.makedirs(out_dir, exist_ok=True)
                return out_dir

        return None

    def _save_result_json(self, result: dict, explainer_name: str):
        out_dir = self._get_output_dir(result["model_type"])
        if not out_dir:
            return

        filename = f"{explainer_name}.json"
        path = os.path.join(out_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        if self.logger:
            self.logger.info(f"Saved ALE result to {path}")
