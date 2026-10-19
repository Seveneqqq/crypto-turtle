import os
import pandas as pd
import joblib
import pickle
import torch
import xgboost as xgb
import numpy as np
from sklearn.pipeline import Pipeline

from xai.explainers.shap_explainer import ShapKernelExplainer, ShapTreeExplainer
from xai.explainers.lime_explainer import LimeExplainer
from xai.explainers.ale_explainer import AleExplainer
from xai.explainers.counterfactuals import CounterfactualExplainer
from xai.explainers.pdp_explainer import PdpExplainer
from xai.explainers.surrogate import SurrogateExplainer

from xai.LSTMPriceModel.model_package import LSTMPriceModel
from xai.utils.logger import setup_logger


class XaiService:
    """
    XaiService:
    - ładuje model
    - waliduje dane
    - wywołuje explainery
    """

    def __init__(self, device="cpu"):
        self.device = device

        PROJECT_ROOT = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

        self.logger = setup_logger(
            name="XaiService",
            base_dir=PROJECT_ROOT
        )

        self.logger.info(f"XaiService initialized (device={self.device})")

    # ======================================================
    # MODEL LOADING
    # ======================================================
    def load_model(self, model_folder, model_file):
        PROJECT_ROOT = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

        model_path = os.path.join(
            PROJECT_ROOT,
            "shared",
            "models",
            model_folder,
            model_file
        )

        self.logger.info(f"Loading model: {model_path}")

        if not os.path.exists(model_path):
            self.logger.error("Model file not found")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        ext = os.path.splitext(model_file)[1].lower()

        # =========================
        # SKLEARN
        # =========================
        if ext in [".pkl", ".joblib"]:
            self.logger.info("Detected sklearn model")
            try:
                model = joblib.load(model_path)
            except Exception:
                self.logger.warning("joblib failed, trying pickle")
                with open(model_path, "rb") as f:
                    model = pickle.load(f)

            return model, "sklearn"

        # =========================
        # LSTM
        # =========================
        elif ext == ".pth":
            self.logger.info("Detected PyTorch LSTM model")

            model_dir = os.path.dirname(model_path)
            config_path = os.path.join(model_dir, "model_config.pkl")

            if not os.path.exists(config_path):
                self.logger.error("model_config.pkl missing")
                raise FileNotFoundError(
                    f"model_config.pkl not found next to LSTM model: {config_path}"
                )

            model_config = joblib.load(config_path)

            model = LSTMPriceModel(
                input_size=len(model_config["features"]),
                hidden_size=model_config["hidden_size"],
                num_layers=model_config["num_layers"],
                dropout=model_config["dropout"]
            ).to(self.device)

            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)

            model.eval()
            model.model_config = model_config

            return model, "lstm"

        # =========================
        # XGBOOST
        # =========================
        elif ext in [".json", ".bin"]:
            self.logger.info("Detected XGBoost model")
            model = xgb.Booster()
            model.load_model(model_path)
            return model, "xgboost"

        else:
            self.logger.error("Unsupported model type")
            raise ValueError(f"Unsupported model type: {ext}")

    # ======================================================
    # MAIN EXPLAIN
    # ======================================================
    def explain_model(
        self,
        rabbit_msg_id,
        model_folder,
        model_file,
        explainer_type,
        data_folder,
        data_file,
        instance=None,
        feature=None,
        explainer_to_test=None
    ):
        self.logger.info("Starting explain_model")
        self.logger.info(f"rabbit_msg_id={rabbit_msg_id}")

        EXPLAINER_REGISTRY = {
            "kernel": ShapKernelExplainer,
            "tree": ShapTreeExplainer,
            "lime": LimeExplainer,
            "ale": AleExplainer,
            "counterfactual": CounterfactualExplainer,
            "pdp": PdpExplainer,
            "surrogate": SurrogateExplainer,
        }

        try:
            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )

            data_path = os.path.join(
                base_dir,
                "shared",
                "files",
                data_folder,
                data_file
            )

            self.logger.info(f"Resolved data path: {data_path}")

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")

            data_ext = os.path.splitext(data_path)[1].lower()

            if data_ext == ".npy":
                X_data = np.load(data_path)
                df = None
                self.logger.info(f"Loaded NPY data, shape={X_data.shape}")
            else:
                df = pd.read_csv(data_path)
                X_data = None
                self.logger.info(f"Loaded CSV data, shape={df.shape}")

            model, model_type = self.load_model(model_folder, model_file)
            self.logger.info(f"Model loaded (type={model_type})")

            explainer_cls = EXPLAINER_REGISTRY.get(explainer_type.lower())
            if not explainer_cls:
                raise ValueError(f"Unknown explainer type: {explainer_type}")

            if explainer_type.lower() == "tree" and model_type == "lstm":
                raise ValueError("TreeExplainer cannot be used with LSTM models")

            explainer = explainer_cls(logger=self.logger)

            # =========================
            # LSTM
            # =========================
            if model_type == "lstm" and X_data is not None:
                result = explainer.explain(
                    model=model,
                    X=X_data,
                    instance=instance
                )

            # =========================
            # SKLEARN / XGBOOST
            # =========================
            else:
                result = explainer.explain(
                    model=model,
                    data_path=data_path,
                    instance=instance,
                    feature=feature
                )

            self.logger.info("Explanation completed successfully")

            return {
                "result": "success",
                "rabbit_msg_id": rabbit_msg_id,
                "data": {
                    "status": "ok",
                    "explainer_type": explainer_type,
                    "model_type": model_type,
                    "instance": instance,
                    "feature": feature,
                    "shap_output": result
                }
            }

        except Exception as e:
            self.logger.exception("ERROR during explanation")
            return {
                "result": "failed",
                "rabbit_msg_id": rabbit_msg_id,
                "message": str(e)
            }
