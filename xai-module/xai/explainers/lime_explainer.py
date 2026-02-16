import numpy as np
import pandas as pd
import os
import json
import traceback
import torch

from lime.lime_tabular import LimeTabularExplainer


class LimeExplainer:
    """
    Unified LIME Explainer
    - sklearn (CSV)
    - LSTM (NPY, 3D sequences)
    Zwraca IDENTYCZNY kontrakt jak SHAP
    """

    def __init__(self, logger=None):
        self.logger = logger
        if self.logger:
            self.logger.info("LimeExplainer initialized")

    # ======================================================
    # MAIN EXPLAIN
    # ======================================================
    def explain(
        self,
        model,
        data_path: str = None,
        X=None,
        instance: int = 0,
        feature: str = None,
        background_size: int = 30,
        nsamples: int = 500
    ):
        if self.logger:
            self.logger.info("Starting LIME explanation")
            self.logger.info(
                f"params: instance={instance}, "
                f"background_size={background_size}, nsamples={nsamples}"
            )

        is_lstm = isinstance(model, torch.nn.Module) and X is not None
        model_type = "lstm" if is_lstm else "sklearn"

        # ======================================================
        # LSTM + NPY
        # ======================================================
        if is_lstm:
            if self.logger:
                self.logger.info("Detected LSTM + NPY data")

            if not hasattr(model, "model_config"):
                self._log_error("model_config missing in LSTM model")
                raise RuntimeError("model_config missing in LSTM model")

            SEQ_LEN = model.model_config["lookback"]
            FEATURES = model.model_config["features"]
            N_FEATURES = len(FEATURES)

            if self.logger:
                self.logger.info(
                    f"LSTM config: seq_len={SEQ_LEN}, n_features={N_FEATURES}"
                )

            if instance < 0 or instance >= X.shape[0]:
                raise ValueError(f"Invalid instance index: {instance}")

            background = X[:background_size].reshape(background_size, -1)
            row = X[instance].reshape(1, -1)

            feature_names = [
                f"{feat}_t-{t}"
                for t in range(SEQ_LEN - 1, -1, -1)
                for feat in FEATURES
            ]

            def predict_fn(x_flat):
                x = torch.tensor(x_flat, dtype=torch.float32)
                x = x.view(x.shape[0], SEQ_LEN, N_FEATURES)
                with torch.no_grad():
                    y = model(x).cpu().numpy().reshape(-1)
                return np.column_stack([1 - y, y])

            lime_mode = "classification"

        # ======================================================
        # SKLEARN + CSV
        # ======================================================
        else:
            if data_path is None:
                raise ValueError("data_path is required for non-NPY explanation")

            if self.logger:
                self.logger.info("Detected sklearn / tabular data")

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")

            df = pd.read_csv(data_path)

            if not hasattr(model, "feature_names_in_"):
                raise RuntimeError("Model must expose feature_names_in_")

            feature_names = list(model.feature_names_in_)
            df = df[feature_names]

            df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
            data = df.to_numpy()

            if instance < 0 or instance >= data.shape[0]:
                raise ValueError(f"Invalid instance index: {instance}")

            background = data[:background_size]
            row = data[instance:instance + 1]

            if hasattr(model, "predict_proba"):
                def predict_fn(x):
                    return model.predict_proba(
                        pd.DataFrame(x, columns=feature_names)
                    )
                lime_mode = "classification"

            elif hasattr(model, "decision_function"):
                def predict_fn(x):
                    scores = model.decision_function(
                        pd.DataFrame(x, columns=feature_names)
                    )
                    probs = 1 / (1 + np.exp(-scores))
                    return np.column_stack([1 - probs, probs])
                lime_mode = "classification"

            else:
                def predict_fn(x):
                    return model.predict(
                        pd.DataFrame(x, columns=feature_names)
                    )
                lime_mode = "regression"

        # ======================================================
        # LIME CORE
        # ======================================================
        try:
            if self.logger:
                self.logger.info(
                    f"Initializing LimeTabularExplainer (mode={lime_mode})"
                )

            explainer = LimeTabularExplainer(
                training_data=background,
                feature_names=feature_names,
                mode=lime_mode,
                discretize_continuous=False
            )

            explanation = explainer.explain_instance(
                row[0],
                predict_fn,
                num_features=len(feature_names),
                num_samples=nsamples
            )

        except Exception:
            if self.logger:
                self.logger.exception("LIME explanation failed")
            traceback.print_exc()
            raise

        # ======================================================
        # MAP LIME → SHAP CONTRACT
        # ======================================================
        lime_map = dict(explanation.as_list())

        shap_values = []
        for feat in feature_names:
            val = 0.0
            for rule, weight in lime_map.items():
                if feat in rule:
                    val = weight
                    break
            shap_values.append(val)

        expected_value = float(
            predict_fn(background).mean(axis=0)[-1]
            if lime_mode == "classification"
            else predict_fn(background).mean()
        )

        result = {
            "instance": instance,
            "model_type": model_type,
            "shap_values": shap_values,
            "expected_value": expected_value,
            "sample_data": row.tolist(),
            "features": feature_names
        }

        # 🔥 LOG + SAVE
        self._log_result_summary(result)
        self._save_result_json(result, explainer_name="lime")

        if self.logger:
            self.logger.info("LIME explanation finished successfully")

        return result

    # ======================================================
    # HELPERS
    # ======================================================
    def _log_error(self, msg):
        if self.logger:
            self.logger.error(msg)

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

        filename = f"{explainer_name}_instance_{result['instance']}.json"
        path = os.path.join(out_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        if self.logger:
            self.logger.info(f"Saved explanation result to {path}")

    def _log_result_summary(self, result: dict):
        if not self.logger:
            return

        shap_vals = np.array(result["shap_values"])
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]

        self.logger.info("LIME result summary:")
        self.logger.info(f"  instance={result['instance']}")
        self.logger.info(f"  expected_value={result['expected_value']:.6f}")
        self.logger.info("  top_features:")

        for i in top_idx:
            self.logger.info(
                f"    {result['features'][i]}: {shap_vals[i]:.6f}"
            )
