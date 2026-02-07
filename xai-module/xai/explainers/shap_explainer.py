import shap
import numpy as np
import pandas as pd
import os
import traceback
import torch
from xai.LSTMPriceModel.model_package import LSTMPriceModel


class ShapKernelExplainer:
    """
    Generic SHAP Kernel Explainer.
    - sklearn (tabular, CSV)
    - LSTM (PyTorch, NPY / sekwencje)
    - liczy SHAP tylko dla jednej instancji
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
        data_path: str = None,   # 🔥 NIE JEST JUŻ WYMAGANE
        X=None,                  # 🔥 NOWE – NPY / LSTM
        instance: int = 0,
        feature: str = None,
        background_size: int = 30,
        nsamples: int = 200
    ):
        print("[ShapKernelExplainer] Starting SHAP Kernel explanation")
        print(f"[ShapKernelExplainer] instance={instance}")

        is_lstm = isinstance(model, torch.nn.Module) and X is not None

        # ======================================================
        # 🔥 ŚCIEŻKA 1: LSTM + NPY (X_test_3d.npy)
        # ======================================================
        if is_lstm and X is not None:
            print("[ShapKernelExplainer] LSTM + NPY data detected")

            if not hasattr(model, "model_config"):
                raise RuntimeError("model_config missing in LSTM model")

            SEQ_LEN = model.model_config["lookback"]
            FEATURES = model.model_config["features"]
            N_FEATURES = len(FEATURES)

            background_size = 10
            nsamples = 50

            if instance < 0 or instance >= X.shape[0]:
                raise ValueError(f"Invalid instance index: {instance}")

            # background: spłaszczone sekwencje
            background = X[:background_size].reshape(background_size, -1)
            print(f"[ShapKernelExplainer] Background shape={background.shape}")

            def predict_fn(x):
                x = torch.tensor(x, dtype=torch.float32)
                x = x.view(x.shape[0], SEQ_LEN, N_FEATURES)
                with torch.no_grad():
                    y = model(x)
                return y.cpu().numpy().ravel()

            row = X[instance:instance + 1].reshape(1, -1)

        # ======================================================
        # 🔥 ŚCIEŻKA 2: klasyczny CSV (sklearn)
        # ======================================================
        else:
            if data_path is None:
                raise ValueError("data_path is required for non-NPY explanation")

            print("[ShapKernelExplainer] Sklearn / tabular data detected")

            df = self._load_data(data_path)

            train_cols = model.feature_names_in_
            df_filtered = df[train_cols].copy()

            for col in df_filtered.columns:
                df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")

            df_filtered = df_filtered.fillna(0)
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

        # ======================================================
        # SHAP
        # ======================================================
        print("[ShapKernelExplainer] Initializing KernelExplainer")
        try:
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(row, nsamples=nsamples)
        except Exception:
            traceback.print_exc()
            raise

        print("[ShapKernelExplainer] SHAP calculation finished")

        return {
            "instance": instance,
            "model_type": "lstm" if is_lstm else "sklearn",
            "shap_values": (
                shap_values.tolist()
                if isinstance(shap_values, np.ndarray)
                else shap_values
            ),
            "expected_value": explainer.expected_value,
            "sample_data": row.tolist(),
            "features": (
                [
                    f"{feat}_t-{t}"
                    for t in range(SEQ_LEN - 1, -1, -1)
                    for feat in FEATURES
                ]
                if is_lstm
                else list(train_cols)
            )
        }


class ShapTreeExplainer:
    """RandomForest / XGBoost / LightGBM"""

    def __init__(self):
        print("[ShapTreeExplainer] Initialized")

    def _load_data(self, data_path: str):
        df = pd.read_csv(data_path)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.fillna(0).to_numpy()

    def explain(self, model, data_path: str):
        data = self._load_data(data_path)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)

        return {
            "shap_values": (
                shap_values.tolist()
                if isinstance(shap_values, np.ndarray)
                else shap_values
            ),
            "expected_value": explainer.expected_value,
            "sample_data": data.tolist()
        }
