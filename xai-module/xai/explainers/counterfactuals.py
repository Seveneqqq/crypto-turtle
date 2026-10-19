import os
import json
import time
import numpy as np
import pandas as pd
import torch


class CounterfactualExplainer:

    def __init__(self, logger=None):
        self.logger = logger
        if self.logger:
            self.logger.info("CounterfactualExplainer initialized")

    # ======================================================
    # MAIN
    # ======================================================
    def explain(
        self,
        model,
        data_path: str = None,
        X=None,
        instance: int = 0,
        target_class: int = 1,
        threshold: float = 0.5,
        max_iter: int = 150,          
        step: float = 0.2,
        background_size: int = 200,
        top_k: int = 6,              
        **kwargs
    ):

        start_time = time.time()

        is_lstm = isinstance(model, torch.nn.Module) and X is not None
        model_type = "lstm" if is_lstm else "sklearn"

        if self.logger:
            self.logger.info(f"Starting Counterfactual ({model_type})")
            self.logger.info(f"instance={instance}, max_iter={max_iter}, top_k={top_k}")

        # ======================================================
        # LSTM
        # ======================================================
        if is_lstm:

            SEQ_LEN = model.model_config["lookback"]
            FEATURES = model.model_config["features"]
            N_FEATURES = len(FEATURES)

            x0 = X[instance].copy()
            flat_x0 = x0.reshape(-1)

            background = X[:background_size]
            bg_flat = background.reshape(background.shape[0], -1)

            f_min = np.min(bg_flat, axis=0)
            f_max = np.max(bg_flat, axis=0)

            def predict_fn(flat):
                t = torch.tensor(flat, dtype=torch.float32)
                t = t.view(t.shape[0], SEQ_LEN, N_FEATURES)
                with torch.no_grad():
                    y = model(t).cpu().numpy().reshape(-1)
                return y

            old_pred = float(predict_fn(flat_x0.reshape(1, -1))[0])

            if self.logger:
                self.logger.info(f"Initial prediction: {old_pred:.6f}")

            def meets_target(p):
                return p >= threshold if target_class == 1 else p < threshold

            cf = flat_x0.copy()

        
            feature_importance_proxy = np.abs(cf - np.mean(bg_flat, axis=0))
            important_idx = np.argsort(feature_importance_proxy)[::-1][:top_k]

            for i in range(max_iter):

                if self.logger and i % 10 == 0:
                    self.logger.info(f"LSTM iteration {i}/{max_iter}")

                improved = False

                for j in important_idx:

                    rng = f_max[j] - f_min[j]
                    if rng == 0:
                        continue

                    delta_val = step * rng

                    for direction in [+1, -1]:
                        candidate = cf.copy()
                        candidate[j] = np.clip(
                            candidate[j] + direction * delta_val,
                            f_min[j],
                            f_max[j]
                        )

                        new_pred = float(
                            predict_fn(candidate.reshape(1, -1))[0]
                        )

                        if meets_target(new_pred):
                            cf = candidate
                            improved = True
                            break

                    if improved:
                        break

                if improved:
                    if self.logger:
                        self.logger.info(f"Target reached at iteration {i}")
                    break

            new_pred = float(predict_fn(cf.reshape(1, -1))[0])

            delta = (cf - flat_x0).tolist()

            feature_names = [
                f"{feat}_t-{t}"
                for t in range(SEQ_LEN - 1, -1, -1)
                for feat in FEATURES
            ]

            expected_value = float(np.mean(predict_fn(bg_flat)))

        # ======================================================
        # SKLEARN
        # ======================================================
        else:

            df = pd.read_csv(data_path)

            if hasattr(model, "feature_names_in_"):
                feature_names = list(model.feature_names_in_)
                df = df[feature_names]
            else:
                feature_names = list(df.columns)

            df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
            data = df.to_numpy()

            x0 = data[instance].copy()
            bg = data[:background_size]

            f_min = np.min(bg, axis=0)
            f_max = np.max(bg, axis=0)

            def predict_fn(arr):
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(arr)[:, 1]
                return model.predict(arr)

            old_pred = float(predict_fn(x0.reshape(1, -1))[0])

            if self.logger:
                self.logger.info(f"Initial prediction: {old_pred:.6f}")

            def meets_target(p):
                return p >= threshold if target_class == 1 else p < threshold

            cf = x0.copy()

            feature_importance_proxy = np.abs(cf - np.mean(bg, axis=0))
            important_idx = np.argsort(feature_importance_proxy)[::-1][:top_k]

            for i in range(max_iter):

                if self.logger and i % 10 == 0:
                    self.logger.info(f"Sklearn iteration {i}/{max_iter}")

                improved = False

                for j in important_idx:

                    rng = f_max[j] - f_min[j]
                    if rng == 0:
                        continue

                    delta_val = step * rng

                    for direction in [+1, -1]:
                        candidate = cf.copy()
                        candidate[j] = np.clip(
                            candidate[j] + direction * delta_val,
                            f_min[j],
                            f_max[j]
                        )

                        new_pred = float(
                            predict_fn(candidate.reshape(1, -1))[0]
                        )

                        if meets_target(new_pred):
                            cf = candidate
                            improved = True
                            break

                    if improved:
                        break

                if improved:
                    if self.logger:
                        self.logger.info(f"Target reached at iteration {i}")
                    break

            new_pred = float(predict_fn(cf.reshape(1, -1))[0])

            delta = (cf - x0).tolist()
            expected_value = float(np.mean(predict_fn(bg)))

        # ======================================================
        # RESULT
        # ======================================================

        result = {
            "instance": instance,
            "model_type": model_type,
            "shap_values": delta,
            "expected_value": expected_value,
            "sample_data": x0.tolist(),
            "features": feature_names,
            "counterfactual": cf.tolist(),
            "old_prediction": old_pred,
            "new_prediction": new_pred
        }

        self._save_result_json(result, explainer_name="counterfactual")

        if self.logger:
            self.logger.info(
                f"Counterfactual finished in {time.time() - start_time:.2f} seconds"
            )

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
            self.logger.info(f"Saved Counterfactual result to {path}")
