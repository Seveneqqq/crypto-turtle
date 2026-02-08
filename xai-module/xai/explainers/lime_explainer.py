import numpy as np
import pandas as pd
import os
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

    def __init__(self):
        print("[LimeExplainer] Initialized")

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
        print("[LimeExplainer] Starting LIME explanation")

        is_lstm = isinstance(model, torch.nn.Module) and X is not None

        # ======================================================
        # ŚCIEŻKA 1: LSTM + NPY
        # ======================================================
        if is_lstm:
            print("[LimeExplainer] LSTM + NPY data detected")

            if not hasattr(model, "model_config"):
                raise RuntimeError("model_config missing in LSTM model")

            SEQ_LEN = model.model_config["lookback"]
            FEATURES = model.model_config["features"]
            N_FEATURES = len(FEATURES)

            if instance < 0 or instance >= X.shape[0]:
                raise ValueError(f"Invalid instance index: {instance}")

            background = X[:background_size].reshape(background_size, -1)
            row = X[instance].reshape(1, -1)

            feature_names = [
                f"{feat}_t-{t}"
                for t in range(SEQ_LEN - 1, -1, -1)
                for feat in FEATURES
            ]

            # ---- predict_fn (LSTM) ----
            def predict_fn(x_flat):
                x = torch.tensor(x_flat, dtype=torch.float32)
                x = x.view(x.shape[0], SEQ_LEN, N_FEATURES)
                with torch.no_grad():
                    y = model(x).cpu().numpy().reshape(-1)
                # classification-style output for LIME
                return np.column_stack([1 - y, y])

            lime_mode = "classification"

        # ======================================================
        # ŚCIEŻKA 2: sklearn / CSV
        # ======================================================
        else:
            if data_path is None:
                raise ValueError("data_path is required for non-NPY explanation")

            print("[LimeExplainer] Sklearn / tabular data detected")

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")

            df = pd.read_csv(data_path)

            if not hasattr(model, "feature_names_in_"):
                raise RuntimeError("Model must expose feature_names_in_")

            feature_names = list(model.feature_names_in_)
            df = df[feature_names]

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.fillna(0)
            data = df.to_numpy()

            if instance < 0 or instance >= data.shape[0]:
                raise ValueError(f"Invalid instance index: {instance}")

            background = data[:background_size]
            row = data[instance:instance + 1]

            # ---- predict_fn (sklearn) ----
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
        # LIME
        # ======================================================
        try:
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

        print("[LimeExplainer] Explanation finished")

        return {
            "instance": instance,
            "model_type": "lstm" if is_lstm else "sklearn",
            "shap_values": shap_values,
            "expected_value": expected_value,
            "sample_data": row.tolist(),
            "features": feature_names
        }
