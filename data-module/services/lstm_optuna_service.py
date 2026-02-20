import os
import json
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, List

import optuna
import numpy as np

from .data_lstm_prepare_service import DataLSTMPrepareService

LSTM_FEATURES = [
    "ma_5", "ma_10", "ma_20",
    "sma_10", "sma_30",
    "cumvol", "volume_norm", "std_20",
    "RSI_14", "EMA_20", "EMA_50",
    "BB_upper", "BB_lower",
    "Momentum_10", "ROC_10",
    "ATR_14", "VWAP", "ROC_5"
]

def generate_combinations(features: List[str], max_size: int) -> List[List[str]]:
    combos = []
    for size in range(1, max_size + 1):
        combos.extend([list(c) for c in combinations(features, size)])
    return combos


class LSTMOptunaService:

    def __init__(self, storage_url: str | None = None):
        self.storage_url = storage_url
        self.logs_dir = os.path.join(os.getcwd(), "shared", "files", "processing_optuna_lstm_log")
        os.makedirs(self.logs_dir, exist_ok=True)

    def run_optimization(self, request: Dict[str, Any]) -> Dict[str, Any]:
        filenames = request["filename_candidates"]
        n_trials = request.get("n_trials", 50)
        steps = request["steps"]
        preprocessing_methods = request.get("preprocessing_methods", ["RAW", "DIFF"])
        seq_len_min, seq_len_max, seq_len_step = request.get("seq_len_range", [32, 128, 32])
        max_features = request.get("max_features_in_combo", 6)
        top_k = request.get("top_k_combinations", 5)

        feature_combinations = generate_combinations(LSTM_FEATURES, max_features)
        study = optuna.create_study(direction="minimize")

        trials_full: List[Dict[str, Any]] = []

        def objective(trial: optuna.Trial) -> float:
            filename = trial.suggest_categorical("filename", filenames)
            preprocessing = trial.suggest_categorical("preprocessing", preprocessing_methods)
            seq_len = trial.suggest_int("seq_len", seq_len_min, seq_len_max, step=seq_len_step)
            feature_combo = trial.suggest_categorical("feature_combo", feature_combinations)

            prep = DataLSTMPrepareService(filename=filename, seq_len=seq_len)
            result = prep.run_pipeline(
                steps=steps,
                preprocessing_method=preprocessing,
                output_folders=["trainmodel_lstm"],
                technical_indicators_set="set_all",
                selected_features=feature_combo,
            )

            # symulacja loss
            val_loss = np.random.rand()

            trials_full.append({
                "trial_number": trial.number,
                "filename": filename,
                "preprocessing": preprocessing,
                "seq_len": seq_len,
                "feature_combo": feature_combo,
                "val_loss": float(val_loss),
                "processed_data_info": result["trainmodel_lstm"],
            })

            return val_loss

        study.optimize(objective, n_trials=n_trials)

        best_trials = sorted(trials_full, key=lambda x: x["val_loss"])[:top_k]

        final_result = {
            "rabbit_msg_id": request.get("rabbit_msg_id"),
            "result": "success",
            "best_value": min(t["val_loss"] for t in best_trials),
            "top_k_feature_combinations": best_trials,
        }

        self._save_log(final_result)
        return final_result

    def _save_log(self, payload: Dict[str, Any]):
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.logs_dir, f"lstm_optuna_{ts}.log")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[LSTM] Log zapisany w {path}")
