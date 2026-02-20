import json
import os
from datetime import datetime
from itertools import combinations

import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .data_service import DataService

FEATURE_POOL = [
    "ret1",
    "ret1_lag1","ret1_lag2","ret1_lag5","ret1_lag10",
    "roll_mean_10","roll_std_10","roll_mean_50","roll_std_50",
    "macd","macd_sig",
    "ma_5","ma_10","ma_20",
    "RSI_14",
    "EMA_20","EMA_50"
]

def generate_combinations(features, max_size):
    out = []
    for k in range(1, max_size+1):
        out += list(combinations(features, k))
    return [tuple(x) for x in out]

class OptunaService:

    def __init__(self):
        self.log_dir = "shared/files/processing_optuna_log"
        os.makedirs(self.log_dir, exist_ok=True)

    def run_optimization(self, request: dict):
        feature_combos = generate_combinations(FEATURE_POOL, request.get("max_features_in_combo",5))
        trials_log = []

        def objective(trial: optuna.Trial):
            feature_combo = trial.suggest_categorical("feature_combo", feature_combos)
            filename = trial.suggest_categorical("filename", request["filename_candidates"])
            horizon = trial.suggest_int("horizon",5,60,step=5)
            preprocessing = trial.suggest_categorical("preprocessing", request["preprocessing_methods"])

            ds = DataService()
            csv_path = ds.generate_and_save(
                filename=filename,
                horizon=horizon,
                output_folder=request["output_folders"][0],
                steps=request["steps"],
                preprocessing=preprocessing,
                feature_combo=list(feature_combo)
            )

            df = pd.read_csv(csv_path).dropna(subset=["target"])

            # tylko feature numeryczne
            feature_cols = [c for c in feature_combo if c in df.columns]
            X = df[feature_cols]
            y = df["target"].astype(int)

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_val, model.predict(X_val))

            trials_log.append({
                "trial": trial.number,
                "accuracy": acc,
                "features": feature_combo,
                "csv": csv_path,
                "preprocessing": preprocessing,
                "horizon": horizon,
                "filename": filename
            })
            return acc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=request["n_trials"])

        best = sorted(trials_log, key=lambda x:x["accuracy"], reverse=True)
        result = {
            "best_accuracy": study.best_value,
            "top_trials": best[:request.get("top_k_combinations",5)]
        }

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.log_dir,f"optuna_{ts}.json")
        with open(log_path,"w") as f:
            json.dump(result,f,indent=2)

        return result
