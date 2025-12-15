import json
import os
from datetime import datetime
from typing import Any, Dict

import optuna
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

from .data_lstm_prepare_service import DataLSTMPrepareService


class LSTMOptunaService:
    def __init__(self, storage_url: str | None = None):
        self.storage_url = storage_url
        self.logs_dir = os.path.join(
            os.getcwd(),
            "shared",
            "files",
            "processing_optuna_lstm_log",
        )
        os.makedirs(self.logs_dir, exist_ok=True)

    def run_optimization(self, request: Dict[str, Any]) -> Dict[str, Any]:

        print("LSTMOptunaService: Starting optimization with request:", request)

        steps = request["steps"]
        filenames = request["filename_candidates"]
        n_trials = request.get("n_trials", 20)

        technical_sets = request.get("technical_sets", ["set1", "set2", "set3", "set4"])
        preprocessing_methods = request.get("preprocessing_methods", ["RAW", "DIFF", "LOG"])
        target_col = request.get("target_col", "close")

        seq_len_min, seq_len_max, seq_len_step = request.get("seq_len_range", [16, 128, 16])
        scaler_types = request.get("scaler_types", ["standard", "minmax"])

        lstm_units_min, lstm_units_max, lstm_units_step = request.get(
            "lstm_units_range",
            [32, 128, 32],
        )
        lstm_layers_min, lstm_layers_max = request.get("lstm_layers_range", [1, 3])
        dropout_min, dropout_max, dropout_step = request.get(
            "dropout_range",
            [0.0, 0.5, 0.1],
        )
        lr_min, lr_max = request.get("lr_range", [1e-4, 1e-2])
        batch_sizes = request.get("batch_sizes", [32, 64, 128])
        epochs = request.get("epochs", 5)

        storage = (
            optuna.storages.RDBStorage(self.storage_url)
            if self.storage_url
            else None
        )

        study = optuna.create_study(
            direction="minimize",
            storage=storage,
            load_if_exists=bool(storage),
        )

        all_trials_full = []

        def objective(trial: optuna.Trial) -> float:
            filename = trial.suggest_categorical("filename", filenames)
            technical_set = trial.suggest_categorical("technical_set", technical_sets)
            preprocessing = trial.suggest_categorical("preprocessing", preprocessing_methods)

            seq_len = trial.suggest_int("seq_len", seq_len_min, seq_len_max, step=seq_len_step)
            scaler_type = trial.suggest_categorical("scaler_type", scaler_types)

            print(f"[Trial {trial.number}] Params: filename={filename}, tech_set={technical_set}, preprocessing={preprocessing}, seq_len={seq_len}, scaler={scaler_type}")

            lstm_units = trial.suggest_int("lstm_units", lstm_units_min, lstm_units_max, step=lstm_units_step)
            lstm_layers = trial.suggest_int("lstm_layers", lstm_layers_min, lstm_layers_max)
            dropout = trial.suggest_float("dropout", dropout_min, dropout_max, step=dropout_step)
            lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
            batch_size = trial.suggest_categorical("batch_size", batch_sizes)

            print(f"[Trial {trial.number}] Model: lstm_units={lstm_units}, lstm_layers={lstm_layers}, dropout={dropout}, lr={lr}, batch_size={batch_size}")

            prep = DataLSTMPrepareService(filename=filename, seq_len=seq_len, scaler_type=scaler_type)
            output_folders = ["trainmodel_lstm"]

            results = prep.run_pipeline(
                steps=steps,
                preprocessing_method=preprocessing,
                output_folders=output_folders,
                technical_indicators_set=technical_set
            )

            info = results["trainmodel_lstm"]
            folder = info["folder"]
            base_name = info["filename"]

            X_train = np.load(os.path.join(folder, f"X_train_{base_name}.npy"))
            y_train = np.load(os.path.join(folder, f"y_train_{base_name}.npy"))
            X_test = np.load(os.path.join(folder, f"X_test_{base_name}.npy"))
            y_test = np.load(os.path.join(folder, f"y_test_{base_name}.npy"))

            print(f"[Trial {trial.number}] Data shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")

            val_ratio = 0.2
            n_train = int(len(X_train) * (1 - val_ratio))
            if n_train <= 0 or n_train >= len(X_train):
                print(f"[Trial {trial.number}] Too few training samples: n_train={n_train}")
                return 1e6

            X_tr, X_val = X_train[:n_train], X_train[n_train:]
            y_tr, y_val = y_train[:n_train], y_train[n_train:]

            print(f"[Trial {trial.number}] Training on {len(X_tr)} samples, validating on {len(X_val)} samples")

            model = self._build_lstm_model(
                input_shape=X_tr.shape[1:],
                lstm_units=lstm_units,
                lstm_layers=lstm_layers,
                dropout=dropout,
                lr=lr,
            )

            history = model.fit(
                X_tr,
                y_tr,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
            )

            val_loss = history.history["val_loss"][-1]
            print(f"[Trial {trial.number}] Finished. val_loss={val_loss:.6f}")

            all_trials_full.append({
                "number": trial.number,
                "value": float(val_loss),
                "params": trial.params
            })

            trial.set_user_attr("folder", folder)
            trial.set_user_attr("base_name", base_name)

            return val_loss

        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_trial.params
        best_value = study.best_value

        result = {
            "result": "success",
            "rabbit_msg_id": request.get("rabbit_msg_id"),
            "message": "Successfully tuned LSTM data & model",
            "best_filename": best_params.get("filename"),
            "best_technical_set": best_params.get("technical_set"),
            "best_preprocessing": best_params.get("preprocessing"),
            "best_seq_len": best_params.get("seq_len"),
            "best_scaler_type": best_params.get("scaler_type"),
            "best_lstm_units": best_params.get("lstm_units"),
            "best_lstm_layers": best_params.get("lstm_layers"),
            "best_dropout": best_params.get("dropout"),
            "best_lr": best_params.get("lr"),
            "best_batch_size": best_params.get("batch_size"),
            "best_params": best_params,
            "best_value": best_value,
            "trials_full": all_trials_full,   
        }

        self._save_run_log(result)
        return result

    def _build_lstm_model(
        self,
        input_shape,
        lstm_units: int,
        lstm_layers: int,
        dropout: float,
        lr: float,
    ) -> keras.Model:

        model = keras.Sequential()
        for i in range(lstm_layers):
            return_sequences = i < (lstm_layers - 1)
            if i == 0:
                model.add(
                    layers.LSTM(
                        lstm_units,
                        return_sequences=return_sequences,
                        input_shape=input_shape,
                    )
                )
            else:
                model.add(
                    layers.LSTM(
                        lstm_units,
                        return_sequences=return_sequences,
                    )
                )

            if dropout > 0:
                model.add(layers.Dropout(dropout))

        model.add(layers.Dense(1, activation="linear"))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
            metrics=["mae"],
        )
        return model

    def _save_run_log(self, log_payload: Dict[str, Any]) -> None:
       
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"optuna_lstm_run_{timestamp}.log"
        filepath = os.path.join(self.logs_dir, filename)

        full_trials = log_payload.get("trials_full", [])
        sorted_trials = sorted(full_trials, key=lambda x: x["value"])

        header = {
            "timestamp": timestamp,
            "best_filename": log_payload.get("best_filename"),
            "best_technical_set": log_payload.get("best_technical_set"),
            "best_preprocessing": log_payload.get("best_preprocessing"),
            "best_seq_len": log_payload.get("best_seq_len"),
            "best_scaler_type": log_payload.get("best_scaler_type"),
            "best_lstm_units": log_payload.get("best_lstm_units"),
            "best_lstm_layers": log_payload.get("best_lstm_layers"),
            "best_dropout": log_payload.get("best_dropout"),
            "best_lr": log_payload.get("best_lr"),
            "best_batch_size": log_payload.get("best_batch_size"),
            "best_value": log_payload.get("best_value"),
            "best_params": log_payload.get("best_params"),
        }

        trials_text_lines = []
        for trial in sorted_trials:
            trials_text_lines.append(f"Trial {trial['number']}")
            trials_text_lines.append(f"value: {trial['value']:.12f}")
            trials_text_lines.append("params:")
            for k, v in trial["params"].items():
                trials_text_lines.append(f"  {k}: {v}")
            trials_text_lines.append("-" * 50)

        body = "\n".join(trials_text_lines)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(header, indent=2, ensure_ascii=False))
            f.write("\n\n All trials (sorted best - worst) \n\n")
            f.write(body)
