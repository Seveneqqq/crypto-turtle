import os
import pandas as pd
import joblib
import pickle
import torch
import xgboost as xgb

from xai.explainers.shap_explainer import ShapKernelExplainer, ShapTreeExplainer


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
        base_dir = os.path.dirname(os.path.dirname(__file__))

        model_path = os.path.join(
            base_dir,
            "shared",
            "models",
            model_folder,
            model_file
        )

        print(f"[XaiService] Loading model: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        ext = os.path.splitext(model_file)[1].lower()

        if ext in [".pkl", ".joblib"]:
            print("[XaiService] Loading sklearn model")
            try:
                model = joblib.load(model_path)
            except Exception:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            return model, "sklearn"

        elif ext == ".pth":
            print("[XaiService] Loading PyTorch model")
            model = torch.load(model_path, map_location=self.device)
            model.eval()
            return model, "lstm"

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

        try:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            data_path = os.path.join(base_dir, "shared", "files", data_folder, data_file)
            print(f"[XaiService] Data path resolved: {data_path}")

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")

            df = pd.read_csv(data_path)
            print(f"[XaiService] Data loaded, shape={df.shape}")

            if instance is not None and (instance < 0 or instance >= len(df)):
                raise ValueError(f"Invalid instance index: {instance}")

            if feature is not None and feature not in df.columns:
                raise ValueError(f"Feature '{feature}' not found in dataset")

            model, model_type = self.load_model(model_folder, model_file)
            print(f"[XaiService] Model loaded (type={model_type})")

            # Wybieramy odpowiedni explainer
            if explainer_type.lower() == "kernel":
                explainer = ShapKernelExplainer()
                shap_result = explainer.explain(model=model, data_path=data_path, instance=instance)
            elif explainer_type.lower() == "tree":
                explainer = ShapTreeExplainer()
                shap_result = explainer.explain(model=model, data_path=data_path)
            else:
                raise ValueError(f"Unknown explainer type: {explainer_type}")
            
            print("[XaiService] Explanation completed successfully")
            print(shap_result)

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
