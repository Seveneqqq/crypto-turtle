import shap
import numpy as np
import pandas as pd
import os
import json
import traceback
import torch
from sklearn.pipeline import Pipeline

class ShapKernelExplainer:
    """
    Generic SHAP Kernel Explainer.
    - sklearn (tabular, CSV)
    - LSTM (PyTorch, NPY / sekwencje)
    - liczy SHAP tylko dla jednej instancji
    """

    def __init__(self, logger=None):
        self.logger = logger
        if self.logger:
            self.logger.info("ShapKernelExplainer initialized")

    # ======================================================
    # DATA LOADING
    # ======================================================
    def _load_data(self, data_path: str):
        if self.logger:
            self.logger.info(f"Loading data from: {data_path}")

        if not os.path.exists(data_path):
            if self.logger:
                self.logger.error("Data file not found")
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        df = pd.read_csv(data_path)

        if self.logger:
            self.logger.info(f"Data loaded, shape={df.shape}")

        return df

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
        nsamples: int = 200
    ):
        if self.logger:
            self.logger.info("Starting SHAP Kernel explanation")
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
                if self.logger:
                    self.logger.error("model_config missing in LSTM model")
                raise RuntimeError("model_config missing in LSTM model")

            SEQ_LEN = model.model_config["lookback"]
            FEATURES = model.model_config["features"]
            N_FEATURES = len(FEATURES)

            background_size = min(background_size, 10)
            nsamples = min(nsamples, 50)

            if instance < 0 or instance >= X.shape[0]:
                raise ValueError(f"Invalid instance index: {instance}")

            background = X[:background_size].reshape(background_size, -1)
            row = X[instance:instance + 1].reshape(1, -1)

            if self.logger:
                self.logger.info(f"LSTM background shape={background.shape}")

            def predict_fn(x):
                x = torch.tensor(x, dtype=torch.float32)
                x = x.view(x.shape[0], SEQ_LEN, N_FEATURES)
                with torch.no_grad():
                    y = model(x)
                return y.cpu().numpy().ravel()

            features = [
                f"{feat}_t-{t}"
                for t in range(SEQ_LEN - 1, -1, -1)
                for feat in FEATURES
            ]

        # ======================================================
        # SKLEARN + CSV
        # ======================================================
        else:
            if data_path is None:
                raise ValueError("data_path is required for non-NPY explanation")

            if self.logger:
                self.logger.info("Detected sklearn / tabular data")

            df = self._load_data(data_path)

            train_cols = model.feature_names_in_
            df_filtered = df[train_cols].copy()

            df_filtered = df_filtered.apply(
                pd.to_numeric, errors="coerce"
            ).fillna(0)

            data = df_filtered.to_numpy()

            if instance < 0 or instance >= data.shape[0]:
                raise ValueError(f"Invalid instance index: {instance}")

            background = data[:background_size]
            row = data[instance:instance + 1]

            if hasattr(model, "predict_proba"):
                predict_fn = lambda x: model.predict_proba(
                    pd.DataFrame(x, columns=train_cols)
                )[:, 1]
            else:
                predict_fn = lambda x: model.predict(
                    pd.DataFrame(x, columns=train_cols)
                )

            features = list(train_cols)

            if self.logger:
                self.logger.info(
                    f"CSV background shape={background.shape}, "
                    f"row shape={row.shape}"
                )

        # ======================================================
        # SHAP CORE
        # ======================================================
        try:
            if self.logger:
                self.logger.info("Initializing shap.KernelExplainer")

            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(row, nsamples=nsamples)

        except Exception:
            if self.logger:
                self.logger.exception("SHAP Kernel explanation failed")
            traceback.print_exc()
            raise

        result = {
            "instance": instance,
            "model_type": model_type,
            "shap_values": (
                shap_values.tolist()
                if isinstance(shap_values, np.ndarray)
                else shap_values
            ),
            "expected_value": explainer.expected_value,
            "sample_data": row.tolist(),
            "features": features
        }

        # LOG + SAVE
        self._log_result_summary(result)
        self._save_result_json(result, explainer_name="shap_kernel")

        if self.logger:
            self.logger.info("SHAP Kernel explanation finished successfully")

        return result

    # ======================================================
    # HELPERS
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
            self.logger.info(f"Saved SHAP result to {path}")

    def _log_result_summary(self, result: dict):
        if not self.logger:
            return

        shap_vals = np.array(result["shap_values"]).flatten()
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]

        self.logger.info("SHAP result summary:")
        self.logger.info(f"  instance={result['instance']}")
        self.logger.info(f"  expected_value={result['expected_value']}")
        self.logger.info("  top_features:")

        for i in top_idx:
            self.logger.info(
                f"    {result['features'][i]}: {shap_vals[i]:.6f}"
            )


# =====================================================================


class ShapTreeExplainer:
    """RandomForest / XGBoost / LightGBM"""

    def __init__(self, logger=None):
        self.logger = logger
        if self.logger:
            self.logger.info("ShapTreeExplainer initialized")

    # ======================================================
    # DATA LOADING
    # ======================================================
    def _load_data(self, data_path: str):
        if not data_path:
            raise ValueError("data_path is required for TreeExplainer")

        if self.logger:
            self.logger.info(f"Loading data for TreeExplainer: {data_path}")

        df = pd.read_csv(data_path)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        if self.logger:
            self.logger.info(f"Tree data loaded, shape={df.shape}")

        return df.to_numpy(), list(df.columns)

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
        **kwargs
    ):
        try:
            if isinstance(model, Pipeline):
                if self.logger:
                    self.logger.info("Extracting final estimator from Pipeline")
                model = model.steps[-1][1]

            data, features = self._load_data(data_path)

            if instance < 0 or instance >= data.shape[0]:
                raise ValueError(f"Invalid instance index: {instance}")

            row = data[instance:instance + 1]

            if self.logger:
                self.logger.info("Initializing shap.TreeExplainer")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(row)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        except Exception:
            if self.logger:
                self.logger.exception("SHAP Tree explanation failed")
            raise

        expected_value = explainer.expected_value

        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value.tolist()
        elif isinstance(expected_value, (np.float32, np.float64)):
            expected_value = float(expected_value)
        
        if isinstance(shap_values, np.ndarray):
            shap_values = shap_values.tolist()
        
        result = {
            "instance": instance,
            "model_type": "tree",
            "shap_values": shap_values,
            "expected_value": expected_value,
            "sample_data": row.tolist(),
            "features": features
        }


        self._save_result_json(result, explainer_name="shap_tree")

        if self.logger:
            self.logger.info("SHAP Tree explanation finished successfully")

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
            self.logger.info(f"Saved SHAP Tree result to {path}")