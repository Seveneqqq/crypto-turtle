import json
import os
from datetime import datetime

import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .data_service import DataService


class OptunaService:
    """
    Orkiestruje optymalizację pipeline’u DataService i
    wewnętrznie ocenia wygenerowane zestawy danych.
    """

    def __init__(self, storage_url: str | None = None):
        self.storage_url = storage_url
        self.logs_dir = os.path.join(
            os.getcwd(),
            "shared",
            "files",
            "processing_optuna_log",
        )
        os.makedirs(self.logs_dir, exist_ok=True)

    def run_optimization(self, request: dict) -> dict:
        steps = request["steps"]
        filenames = request["filename_candidates"]
        n_trials = request.get("n_trials", 20)
        tech_sets = request.get("technical_sets", ["set1", "set2", "set3"])
        preprocessing_methods = request.get(
            "preprocessing_methods",
            ["RAW", "Z-SCORE", "ROBUST"],
        )
        output_folders = request.get("output_folders", ["trainmodel"])
        metrics = request.get("metrics", ["accuracy"])

        storage = (
            optuna.storages.RDBStorage(self.storage_url)
            if self.storage_url
            else None
        )
        study = optuna.create_study(
            direction="maximize",
            storage=storage,
            load_if_exists=bool(storage),
        )

        def objective(trial: optuna.Trial) -> float:
            filename = trial.suggest_categorical("filename", filenames)
            horizon = trial.suggest_int("horizon", 5, 60, step=5)
            tech_set = trial.suggest_categorical("technical_set", tech_sets)
            preprocessing = trial.suggest_categorical(
                "preprocessing",
                preprocessing_methods,
            )

            ds = DataService(filename=filename, horizon=horizon)
            processed = ds.run_pipeline(
                steps=steps,
                preprocessing_method=preprocessing,
                output_folders=output_folders,
                technical_indicators_set=tech_set,
            )

            primary_folder = output_folders[0]
            processed_file = processed[primary_folder]["filepath"]
            score = self._evaluate_dataset(processed_file, metrics)
            return score

        study.optimize(objective, n_trials=n_trials)

        result = {
            "result": "success",
            "rabbit_msg_id": request.get("rabbit_msg_id"),
            "message": "Successfully tuned data",
            "best_filename": study.best_trial.params.get("filename"),
            "best_horizon": study.best_trial.params.get("horizon"),
            "best_technical_set": study.best_trial.params.get("technical_set"),
            "best_preprocessing": study.best_trial.params.get("preprocessing"),
            "best_params": study.best_trial.params,
            "best_value": study.best_value,
            "trials_summary": [(t.number, t.value) for t in study.trials],
        }
        self._save_run_log(result)
        return result

    def _save_run_log(self, log_payload: dict) -> None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"optuna_run_{timestamp}.log"
        filepath = os.path.join(self.logs_dir, filename)

        header = {
            "timestamp": timestamp,
            "best_filename": log_payload["best_filename"],
            "best_horizon": log_payload["best_horizon"],
            "best_technical_set": log_payload["best_technical_set"],
            "best_preprocessing": log_payload["best_preprocessing"],
            "best_value": log_payload["best_value"],
            "best_params": log_payload["best_params"],
        }

        body = "\n".join(
            f"Trial {number}: value={value}"
            for number, value in log_payload["trials_summary"]
        ) or "Brak zakończonych prób."

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(header, indent=2))
            f.write("\n\n=== TRIAL LOGS ===\n")
            f.write(body)

    # --- prywatna część oceny ---

    def _evaluate_dataset(self, filepath: str, metrics: list[str]) -> float:
        """
        Wbudowany scorer: wczytuje CSV, trenuje LogisticRegression na cechach
        numerycznych i zwraca accuracy. Gdy będziesz gotowy, podmień to np.
        na backtest bota lub inną metrykę.
        """
        df = pd.read_csv(filepath)

        if "target" not in df.columns:
            raise ValueError("Dataset nie ma kolumny 'target' – brak metryki.")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if "target" not in numeric_cols:
            raise ValueError("Kolumna 'target' nie jest numeryczna.")
        numeric_cols.remove("target")

        if not numeric_cols:
            raise ValueError("Brak numerycznych cech do trenowania modelu.")

        X = df[numeric_cols].replace([float("inf"), float("-inf")], pd.NA)
        y = df["target"].astype(int)

        imputer = SimpleImputer(strategy="median")
        X = pd.DataFrame(
            imputer.fit_transform(X),
            columns=numeric_cols,
            index=df.index,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=False,
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        return accuracy_score(y_val, preds)