import os
import json
import numpy as np
import pandas as pd
import torch


class PdpExplainer:

    def __init__(self, logger=None):
        self.logger = logger
        if self.logger:
            self.logger.info("PdpExplainer initialized")

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
        grid_points: int = 20,
        background_size: int = 300,
        **kwargs
    ):

        is_lstm = isinstance(model, torch.nn.Module) and X is not None
        model_type = "lstm" if is_lstm else "sklearn"

        if self.logger:
            self.logger.info(f"Starting PDP ({model_type})")

        # ======================================================
        # SKLEARN
        # ======================================================
        if not is_lstm:

            df = pd.read_csv(data_path)

            if hasattr(model, "feature_names_in_"):
                feature_names = list(model.feature_names_in_)
                df = df[feature_names]
            else:
                feature_names = list(df.columns)

            df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
            data = df.to_numpy()

            background = data[:background_size]

            def predict_fn(arr):
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(arr)[:, 1]
                return model.predict(arr)

            expected_value = float(np.mean(predict_fn(background)))
            sample_data = data[instance].tolist()

        # ======================================================
        # LSTM
        # ======================================================
        else:

            if not hasattr(model, "model_config"):
                raise RuntimeError("model_config missing in LSTM model")

            SEQ_LEN = model.model_config["lookback"]
            FEATURES = model.model_config["features"]

            feature_names = FEATURES
            background = X[:background_size]

            def predict_fn(arr):
                t = torch.tensor(arr, dtype=torch.float32)
                with torch.no_grad():
                    y = model(t).cpu().numpy().reshape(-1)
                return y

            expected_value = float(np.mean(predict_fn(background)))
            sample_data = X[instance].reshape(-1).tolist()

        # ======================================================
        # WYBÓR CECH
        # ======================================================

        if feature is None:
            features_to_compute = feature_names
        else:
            if feature not in feature_names:
                raise ValueError(f"Feature '{feature}' not found")
            features_to_compute = [feature]

        all_pdp_values = []
        all_grids = []

        # ======================================================
        # LICZENIE PDP
        # ======================================================

        for feat in features_to_compute:

            if self.logger:
                self.logger.info(f"Computing PDP for feature={feat}")

            feature_idx = feature_names.index(feat)

            if not is_lstm:
                f_min = np.min(background[:, feature_idx])
                f_max = np.max(background[:, feature_idx])
                grid = np.linspace(f_min, f_max, grid_points)

                pdp_values = []

                for val in grid:
                    modified = background.copy()
                    modified[:, feature_idx] = val
                    preds = predict_fn(modified)
                    pdp_values.append(float(np.mean(preds)))

            else:
                f_min = np.min(background[:, :, feature_idx])
                f_max = np.max(background[:, :, feature_idx])
                grid = np.linspace(f_min, f_max, grid_points)

                pdp_values = []

                for val in grid:
                    modified = background.copy()
                    modified[:, :, feature_idx] = val
                    preds = predict_fn(modified)
                    pdp_values.append(float(np.mean(preds)))

            all_pdp_values.append(pdp_values)
            all_grids.append(grid.tolist())

        # ======================================================
        # RESULT
        # ======================================================

        result = {
            "instance": instance,
            "model_type": model_type,
            "shap_values": all_pdp_values,
            "expected_value": expected_value,
            "sample_data": sample_data,
            "features": features_to_compute,
            "sample_grid": all_grids
        }

        self._save_result_json(result, explainer_name="pdp")

        if self.logger:
            self.logger.info("PDP explanation finished successfully")

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

        filename = f"{explainer_name}_instance_{result['instance']}.json"
        path = os.path.join(out_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        if self.logger:
            self.logger.info(f"Saved PDP result to {path}")
