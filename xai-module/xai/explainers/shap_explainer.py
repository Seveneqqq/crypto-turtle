import shap
import numpy as np
import pandas as pd
import os
import traceback

class ShapKernelExplainer:
    """
    Generic SHAP Kernel Explainer.
    Model jest przekazywany – nie ładowany z pliku.
    Obsługuje classifiery i regresję.
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

    def explain(self, model, data_path: str, background_size: int = 50):
        print("[ShapKernelExplainer] Starting SHAP Kernel explanation")

        df = self._load_data(data_path)
        train_cols = model.feature_names_in_
        print(f"[ShapKernelExplainer] Selecting model features: {train_cols}")

        df_filtered = df[train_cols].copy()

        # Konwersja kolumn na float i logowanie
        for col in df_filtered.columns:
            before_dtype = df_filtered[col].dtype
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
            nan_count = df_filtered[col].isna().sum()
            print(f"[ShapKernelExplainer] Converting column '{col}' to numeric")
            print(f"    dtype before={before_dtype}, after={df_filtered[col].dtype}, nan_count={nan_count}")

        df_filtered = df_filtered.fillna(0)
        print("[ShapKernelExplainer] Missing values replaced with 0")

        data = df_filtered.to_numpy()
        background = data[:background_size]
        print(f"[ShapKernelExplainer] Background shape={background.shape}")

        # Sprawdzenie, czy model jest classifierem
        if hasattr(model, "predict_proba"):
            print("[ShapKernelExplainer] Detected classifier: using predict_proba[:,1]")
            predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=train_cols))[:, 1]
        else:
            print("[ShapKernelExplainer] Detected regression model: using predict")
            predict_fn = lambda x: model.predict(pd.DataFrame(x, columns=train_cols))

        print("[ShapKernelExplainer] Initializing KernelExplainer")
        try:
            explainer = shap.KernelExplainer(predict_fn, background)
        except Exception as e:
            print(f"[ShapKernelExplainer] FAILED to initialize KernelExplainer: {e}")
            traceback.print_exc()
            raise

        # Obliczanie SHAP z logowaniem crash row
        print("[ShapKernelExplainer] Calculating SHAP values")
        shap_values = None
        for i in range(data.shape[0]):
            try:
                shap_values = explainer.shap_values(data[i:i+1], nsamples="auto")
            except Exception as e:
                print(f"[ShapKernelExplainer] [!!!] SHAP crashed at row {i}")
                print(f"Row data: {data[i]}")
                print(traceback.format_exc())
                raise e

        print("[ShapKernelExplainer] SHAP calculation finished")
        return {
            "shap_values": shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
            "expected_value": explainer.expected_value,
            "sample_data": data.tolist()
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
        print(f"[ShapTreeExplainer] Data loaded, shape={df.shape}")

        # Konwersja danych do float i logowanie
        for col in df.columns:
            before_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors='coerce')
            nan_count = df[col].isna().sum()
            print(f"[ShapTreeExplainer] Converting column '{col}' to numeric")
            print(f"    dtype before={before_dtype}, after={df[col].dtype}, nan_count={nan_count}")
        df = df.fillna(0)
        print("[ShapTreeExplainer] Missing values replaced with 0")

        return df.to_numpy()

    def explain(self, model, data_path: str):
        print("[ShapTreeExplainer] Starting Tree SHAP explanation")
        data = self._load_data(data_path)

        print("[ShapTreeExplainer] Initializing TreeExplainer")
        try:
            explainer = shap.TreeExplainer(model)
        except Exception as e:
            print(f"[ShapTreeExplainer] FAILED to initialize TreeExplainer: {e}")
            traceback.print_exc()
            raise

        print("[ShapTreeExplainer] Calculating SHAP values")
        try:
            shap_values = explainer.shap_values(data)
        except Exception as e:
            print(f"[ShapTreeExplainer] [!!!] SHAP crashed")
            for i in range(data.shape[0]):
                try:
                    explainer.shap_values(data[i:i+1])
                except:
                    print(f"Row {i} causes crash: {data[i]}")
                    break
            raise e

        print("[ShapTreeExplainer] SHAP calculation finished")
        return {
            "shap_values": shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
            "expected_value": explainer.expected_value,
            "sample_data": data.tolist()
        }
