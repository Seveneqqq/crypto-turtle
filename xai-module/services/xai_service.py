import os
import pandas as pd
import joblib
import pickle
import torch
import xgboost as xgb
import numpy as np

from xai.explainers.shap_explainer import ShapKernelExplainer, ShapTreeExplainer
from xai.explainers.lime_explainer import LimeExplainer
from xai.explainers.ale_explainer import AleExplainer
from xai.explainers.counterfactuals import CounterfactualExplainer
from xai.explainers.pdp_explainer import PdpExplainer
from xai.explainers.surrogate import SurrogateExplainer

from xai.LSTMPriceModel.model_package import LSTMPriceModel


class XaiService:
    """
    XaiService:
    - ładuje model
    - waliduje dane
    - wywołuje explainery
    """

    def __init__(self, device="cpu"):
        self.device = device
        print(f"[XaiService] Initialized (device={self.device})")

    
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
    
        print(f"[XaiService] Loading model: {model_path}")
    
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
        ext = os.path.splitext(model_file)[1].lower()
    
        # =========================
        # SKLEARN
        # =========================
        if ext in [".pkl", ".joblib"]:
            print("[XaiService] Loading sklearn model")
            try:
                model = joblib.load(model_path)
            except Exception:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            return model, "sklearn"
    
        # =========================
        # LSTM
        # =========================
        elif ext == ".pth":
            print("[XaiService] Loading PyTorch LSTM model (state_dict)")
    
            model_dir = os.path.dirname(model_path)
            config_path = os.path.join(model_dir, "model_config.pkl")
    
            if not os.path.exists(config_path):
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
            print("[XaiService] Loading XGBoost model")
            model = xgb.Booster()
            model.load_model(model_path)
            return model, "xgboost"
    
        else:
            raise ValueError(f"Unsupported model type: {ext}")


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
        print("[XaiService] Starting explain_model")
        print(f"[XaiService] rabbit_msg_id={rabbit_msg_id}")

        EXPLAINER_REGISTRY = {
            "kernel": ShapKernelExplainer,
            "tree": ShapTreeExplainer,
            "lime": LimeExplainer,
            "ale": AleExplainer,
            "counterfactual": CounterfactualExplainer,
            "pdp": PdpExplainer,
            "surrogate": SurrogateExplainer,
        }

        X_data = None
        df = None

        try:
            base_dir = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__)
                )
            )

            data_path = os.path.join(
                base_dir,
                "shared",
                "files",
                data_folder,
                data_file
            )

            print(f"[XaiService] Data path resolved: {data_path}")

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")

            data_ext = os.path.splitext(data_path)[1].lower()

            # 🔥 NIE ŁADUJ CSV DLA NPY
            if data_ext == ".npy":
                X_data = np.load(data_path)
                df = None
                print(f"[XaiService] Loaded NPY data, shape={X_data.shape}")
            else:
                df = pd.read_csv(data_path)
                X_data = None
                print(f"[XaiService] Data loaded, shape={df.shape}")

            model, model_type = self.load_model(model_folder, model_file)
            print(f"[XaiService] Model loaded (type={model_type})")

            explainer_cls = EXPLAINER_REGISTRY.get(explainer_type.lower())
            if not explainer_cls:
                raise ValueError(f"Unknown explainer type: {explainer_type}")

            explainer = explainer_cls()

            # 🔥 ROZDZIELENIE ŚCIEŻEK
            if model_type == "lstm" and X_data is not None:
                shap_result = explainer.explain(
                    model=model,
                    X=X_data,
                    instance=instance
                )
            else:
                shap_result = explainer.explain(
                    model=model,
                    data_path=data_path,
                    instance=instance,
                    feature=feature
                )

            print("[XaiService] Explanation completed successfully")

            return {
                "result": "success",
                "rabbit_msg_id": rabbit_msg_id,
                "data": {
                    "status": "ok",
                    "explainer_type": explainer_type,
                    "model_type": model_type,
                    "instance": instance,
                    "feature": feature,
                    "shap_output": shap_result
                }
            }

        except Exception as e:
            print("[XaiService] ERROR during explanation")
            print(str(e))
            return {
                "result": "failed",
                "rabbit_msg_id": rabbit_msg_id,
                "message": str(e)
            }
