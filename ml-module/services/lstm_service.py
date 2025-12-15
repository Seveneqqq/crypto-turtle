import os
import sys
from datetime import datetime


class LSTMService:
    def __init__(self, shared_dir: str = None):
        if shared_dir is None:
            shared_dir = self._find_project_root()

        self.SHARED_DIR = shared_dir
        self.MODELS_DIR = os.path.join(shared_dir, "shared", "models")
        self.PREPARED_DATA_DIR = os.path.join(shared_dir, "shared", "files", "lstm_prepared_data")
        self.RAW_DATA_DIR = os.path.join(shared_dir, "shared", "files", "raw_data")

        os.makedirs(os.path.join(self.MODELS_DIR, "lstm_models"), exist_ok=True)
        os.makedirs(os.path.join(self.MODELS_DIR, "lstm_data"), exist_ok=True)
        os.makedirs(self.PREPARED_DATA_DIR, exist_ok=True)

        self.lstm_model_dir = os.path.join(self._find_ml_module_root(), "lstm_model")
        if self.lstm_model_dir not in sys.path:
            sys.path.insert(0, self.lstm_model_dir)

        print(f"[LSTMService] LSTM models dir: {os.path.join(self.MODELS_DIR, 'lstm_models')}")
        print(f"[LSTMService] LSTM data dir: {os.path.join(self.MODELS_DIR, 'lstm_data')}")
        print(f"[LSTMService] Prepared data dir: {self.PREPARED_DATA_DIR}")

    @staticmethod
    def _find_project_root():
        current = os.path.dirname(os.path.abspath(__file__))
        while not os.path.exists(os.path.join(current, "shared")):
            parent = os.path.dirname(current)
            if parent == current:
                raise FileNotFoundError("Cannot find project root with /shared")
            current = parent
        return current

    @staticmethod
    def _find_ml_module_root():
        path = "/app/ml-module"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected ml-module at {path}, but directory does not exist")
        return path

    def _resolve_data_path(self, data_path: str) -> str:
        """Resolve data path - check prepared_data, raw_data, or use as-is"""
        if os.path.isabs(data_path) and os.path.exists(data_path):
            return data_path

        prepared_candidate = os.path.join(self.PREPARED_DATA_DIR, data_path)
        if os.path.exists(prepared_candidate):
            return prepared_candidate

        raw_candidate = os.path.join(self.RAW_DATA_DIR, data_path)
        if os.path.exists(raw_candidate):
            return raw_candidate

        return data_path

    def train_model(self,
                    rabbit_msg_id,
                    train_path: str,
                    lookback: int = 48,
                    horizon: int = 1,
                    hidden_size: int = 128,
                    num_layers: int = 1,
                    dropout: float = 0.2,
                    epochs: int = 100,
                    batch_size: int = 256,
                    lr: float = 0.0005,
                    weight_decay: float = 1e-5,
                    use_subset: bool = False,
                    subset_size: int = 100000,
                    feature_set: str = None,
                    custom_features: list = None,
                    validate_features: bool = True,
                    generate_features: bool = True,
                    scheduler_type: str = 'none',
                    use_log_prices: bool = True,
                    early_stopping_patience: int = 25,
                    load_optuna_params: str = None,
                    load_optuna_dir: str = None,
                    train_all_top: bool = False,
                    seed: int = 42) -> dict:

        try:
            from lstm_train_price import train_model

            original_cwd = os.getcwd()
            os.chdir(os.path.join(self.MODELS_DIR))

            train_path = self._resolve_data_path(train_path)

            if load_optuna_params:
                load_optuna_params = self._resolve_data_path(load_optuna_params)
            if load_optuna_dir:
                load_optuna_dir = self._resolve_data_path(load_optuna_dir)

            result = train_model(
                train_path=train_path,
                lookback=lookback,
                horizon=horizon,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                use_subset=use_subset,
                subset_size=subset_size,
                feature_set=feature_set,
                custom_features=custom_features,
                validate_features_flag=validate_features,
                generate_features=generate_features,
                scheduler_type=scheduler_type,
                use_log_prices=use_log_prices,
                early_stopping_patience=early_stopping_patience,
                load_optuna_params=load_optuna_params,
                load_optuna_dir=load_optuna_dir,
                train_all_top=train_all_top,
                seed=seed
            )

            os.chdir(original_cwd)

            if isinstance(result, list):
                model_dirs = [r[3]['output_dir'] for r in result]
                model_ids = [r[3]['model_id'] for r in result]
                return {
                    "rabbit_msg_id": rabbit_msg_id,
                    "result": "success",
                    "message": f"Successfully trained {len(result)} LSTM models",
                    "model_ids": model_ids,
                    "model_dirs": model_dirs,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                model, feature_scaler, price_scaler, config = result
                return {
                    "rabbit_msg_id": rabbit_msg_id,
                    "result": "success",
                    "message": "Successfully trained LSTM model",
                    "model_id": config['model_id'],
                    "model_dir": config['output_dir'],
                    "best_val_loss": config['best_val_loss'],
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            os.chdir(original_cwd)
            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def test_model(self,
                   rabbit_msg_id,
                   model_dir: str = None,
                   test_path: str = None,
                   batch_size: int = 1024) -> dict:

        try:
            from lstm_test_price import test_model

            original_cwd = os.getcwd()
            os.chdir(os.path.join(self.MODELS_DIR))

            if model_dir:
                if not os.path.isabs(model_dir):
                    model_candidate = os.path.join(self.MODELS_DIR, "lstm_models", model_dir)
                    if os.path.exists(model_candidate):
                        model_dir = model_candidate

            if test_path:
                test_path = self._resolve_data_path(test_path)

            metrics = test_model(
                model_dir=model_dir,
                test_path=test_path,
                batch_size=batch_size
            )

            os.chdir(original_cwd)

            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "success",
                "message": "Successfully tested LSTM model",
                "metrics": {
                    "mae": float(metrics['mae']),
                    "rmse": float(metrics['rmse']),
                    "mape": float(metrics['mape']),
                    "r2": float(metrics['r2']),
                    "dir_acc": float(metrics['dir_acc'])
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            os.chdir(original_cwd)
            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def prepare_data(self,
                     rabbit_msg_id,
                     data_path: str,
                     feature_set: str = 'full',
                     train_months: int = None,
                     agent_months: int = None,
                     test_months: int = None,
                     train_pct: float = 70.0,
                     agent_pct: float = 0.0,
                     test_pct: float = 30.0,
                     lookback: int = 48,
                     horizon: int = 1,
                     scaler_features: str = 'standard',
                     scaler_price: str = 'standard',
                     use_log_prices: bool = True) -> dict:

        try:
            from prepare_data import prepare_data

            original_cwd = os.getcwd()
            os.chdir(self.PREPARED_DATA_DIR)

            data_path = self._resolve_data_path(data_path)

            output_path, data_id = prepare_data(
                data_path=data_path,
                output_dir=self.PREPARED_DATA_DIR,
                feature_set=feature_set,
                train_months=train_months,
                agent_months=agent_months,
                test_months=test_months,
                train_pct=train_pct,
                agent_pct=agent_pct,
                test_pct=test_pct,
                lookback=lookback,
                horizon=horizon,
                scaler_features=scaler_features,
                scaler_price=scaler_price,
                use_log_prices=use_log_prices
            )

            os.chdir(original_cwd)

            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "success",
                "message": "Successfully prepared LSTM data",
                "data_id": data_id,
                "output_path": output_path,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            os.chdir(original_cwd)
            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def tune_hyperparameters(self,
                             rabbit_msg_id,
                             train_path: str,
                             n_trials: int = 100,
                             timeout: int = None,
                             study_name: str = None,
                             subset_size: int = 200000,
                             feature_set: str = None,
                             custom_features: list = None,
                             validate_features: bool = True,
                             generate_features: bool = True,
                             fixed_horizon: int = None) -> dict:

        try:
            import sys
            sys.path.insert(0, self.lstm_model_dir)
            from lstm_hyperparameter_tuning import run_tuning, TuningConfig

            original_cwd = os.getcwd()
            os.chdir(os.path.join(self.MODELS_DIR, "lstm_data"))

            train_path = self._resolve_data_path(train_path)

            TuningConfig.TRAIN_PATH = train_path
            TuningConfig.SUBSET_SIZE = subset_size
            TuningConfig.FEATURE_SET = feature_set
            TuningConfig.CUSTOM_FEATURES = custom_features
            TuningConfig.VALIDATE_FEATURES = validate_features
            TuningConfig.GENERATE_FEATURES = generate_features
            TuningConfig.FIXED_HORIZON = fixed_horizon

            study, best_trial = run_tuning(
                n_trials=n_trials,
                timeout=timeout,
                study_name=study_name
            )

            os.chdir(original_cwd)

            if best_trial is None:
                return {
                    "rabbit_msg_id": rabbit_msg_id,
                    "result": "failed",
                    "message": "No trials completed successfully",
                    "timestamp": datetime.now().isoformat()
                }

            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "success",
                "message": "Successfully completed hyperparameter tuning",
                "best_trial_number": best_trial.number,
                "best_value": float(best_trial.value),
                "best_params": best_trial.params,
                "study_name": study.study_name,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            os.chdir(original_cwd)
            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }