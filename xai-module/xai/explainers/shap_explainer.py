import shap
import numpy as np
import pandas as pd
import os
import traceback


class ShapKernelExplainer:
    """
    Generic SHAP Kernel Explainer.
    - liczy SHAP tylko dla jednej instancji
    - model przekazywany jako obiekt
    """

    def __init__(self):
        print("[ShapKernelExplainer] Initialized")

    def _load_data(self, data_path: str):
        print(f"[ShapKernelExplainer] Loading data from: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        df = pd.read_csv(data_path)
        print(f"[ShapKernelExplainer] Data loaded shape={df.shape}")
        return df

    def explain(
        self,
        model,
        data_path: str,
        instance: int = 0,
        background_size: int = 30,
        nsamples: int = 200
    ):
        print("[ShapKernelExplainer] Starting SHAP Kernel explanation")
        print(f"[ShapKernelExplainer] instance={instance}")

        df = self._load_data(data_path)

        # kolumny zgodne z modelem
        train_cols = model.feature_names_in_
        print(f"[ShapKernelExplainer] Using model features: {train_cols}")

        df_filtered = df[train_cols].copy()

        # konwersja na float
        for col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")

        df_filtered = df_filtered.fillna(0)

        data = df_filtered.to_numpy()

        if instance < 0 or instance >= data.shape[0]:
            raise ValueError(f"Invalid instance index: {instance}")

        # background
        background = data[:background_size]
        print(f"[ShapKernelExplainer] Background shape={background.shape}")

        # wybór funkcji predykcji
        if hasattr(model, "predict_proba"):
            print("[ShapKernelExplainer] Classifier detected")
            predict_fn = lambda x: model.predict_proba(
                pd.DataFrame(x, columns=train_cols)
            )[:, 1]
        else:
            print("[ShapKernelExplainer] Regression model detected")
            predict_fn = lambda x: model.predict(
                pd.DataFrame(x, columns=train_cols)
            )

        # init explainer
        print("[ShapKernelExplainer] Initializing KernelExplainer")
        try:
            explainer = shap.KernelExplainer(predict_fn, background)
        except Exception:
            traceback.print_exc()
            raise

        # SHAP tylko dla jednej instancji
        print("[ShapKernelExplainer] Calculating SHAP values (single instance)")
        try:
            row = data[instance:instance + 1]
            shap_values = explainer.shap_values(row, nsamples=nsamples)
        except Exception:
            traceback.print_exc()
            raise

        print("[ShapKernelExplainer] SHAP calculation finished")

        return {
            "instance": instance,
            "shap_values": (
                shap_values.tolist()
                if isinstance(shap_values, np.ndarray)
                else shap_values
            ),
            "expected_value": explainer.expected_value,
            "sample_data": row.tolist(),
            "features": list(train_cols)
        }
    
class ShapTreeExplainer:
    """RandomForest / XGBoost / LightGBM"""

    def __init__(self):
        print("[ShapTreeExplainer] Initialized")

    def _load_data(self, data_path: str):
        print(f"[ShapTreeExplainer] Loading data from: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        df = pd.read_csv(data_path)

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.fillna(0)
        return df.to_numpy()

    def explain(self, model, data_path: str):
        print("[ShapTreeExplainer] Starting Tree SHAP explanation")

        data = self._load_data(data_path)

        print("[ShapTreeExplainer] Initializing TreeExplainer")
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            traceback.print_exc()
            raise

        print("[ShapTreeExplainer] Calculating SHAP values")
        try:
            shap_values = explainer.shap_values(data)
        except Exception:
            traceback.print_exc()
            raise

        print("[ShapTreeExplainer] SHAP calculation finished")

        return {
            "shap_values": (
                shap_values.tolist()
                if isinstance(shap_values, np.ndarray)
                else shap_values
            ),
            "expected_value": explainer.expected_value,
            "sample_data": data.tolist()
        }
