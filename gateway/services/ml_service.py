from configuration.rabbitmq_queues import ML_REQUEST_QUEUE

class MlService:
    def __init__(self, sqlite_service, rabbitmq_service):
        self.sqlite_service = sqlite_service
        self.rabbitmq_service = rabbitmq_service

    def handle_response(self, message: dict):
        print("Received response from RabbitMQ:")
        print(message)
        rabbit_msg_id = message.get("rabbit_msg_id")
        rabbit_result = message.get("result")
        self.sqlite_service.update_message(
            rabbit_msg_id=rabbit_msg_id,
            state=rabbit_result,
            response=str(message)
        )
        print("Response updated in SQLite")

    def just_send_data_to_module(self, data: dict):
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        action = data.get("action")
        if not action:
            raise ValueError("Missing 'action' in request payload")
        request_payload = {
            "action": action,
            "rabbit_msg_id": rabbit_msg_id,
            "data": data
        }
        self.rabbitmq_service.publish(queue=ML_REQUEST_QUEUE, message=request_payload)
        print("Request with arbitrary data sent to RabbitMQ")
        print(request_payload)
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )

    def train_model(self, df_filename: str, clf_version: str, feature_cols: list = None):
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        request_payload = {
            "action": "train_model",
            "rabbit_msg_id": rabbit_msg_id,
            "df_filename": df_filename,
            "clf_version": clf_version,
            "feature_cols": feature_cols
        }
        self.rabbitmq_service.publish(queue=ML_REQUEST_QUEUE, message=request_payload)
        print("Request for model training sent to RabbitMQ")
        print(request_payload)
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )

    def prepare_features(self, train_file: str, test_file: str, fe_version: str = "fe_v2", feature_cols: list = None):
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        request_payload = {
            "action": "prepare_features",
            "rabbit_msg_id": rabbit_msg_id,
            "train_file": train_file,
            "test_file": test_file,
            "fe_version": fe_version,
            "feature_cols": feature_cols
        }
        self.rabbitmq_service.publish(queue=ML_REQUEST_QUEUE, message=request_payload)
        print("Request for preparing features sent to RabbitMQ")
        print(request_payload)
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )

    def test_model(self, model_filename: str, test_filename: str):
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        request_payload = {
            "action": "test_model",
            "rabbit_msg_id": rabbit_msg_id,
            "model_filename": model_filename,
            "test_filename": test_filename
        }
        self.rabbitmq_service.publish(queue=ML_REQUEST_QUEUE, message=request_payload)
        print("Request for testing model sent to RabbitMQ")
        print(request_payload)
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )

    def tune_model(self, df_train_filename: str, df_test_filename: str, n_iter: int = 50, cv_folds: int = 3,
                   random_seed: int = 42):
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        request_payload = {
            "action": "tune_model",
            "rabbit_msg_id": rabbit_msg_id,
            "df_train_filename": df_train_filename,
            "df_test_filename": df_test_filename,
            "n_iter": n_iter,
            "cv_folds": cv_folds,
            "random_seed": random_seed
        }
        self.rabbitmq_service.publish(queue=ML_REQUEST_QUEUE, message=request_payload)
        print("Request for hyperparameter tuning sent to RabbitMQ")
        print(request_payload)
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )

    def lstm_train(self, train_path: str, lookback: int = 48, horizon: int = 1, hidden_size: int = 128,
                   num_layers: int = 1, dropout: float = 0.2, epochs: int = 100, batch_size: int = 256,
                   lr: float = 0.0005, weight_decay: float = 1e-5, use_subset: bool = False,
                   subset_size: int = 100000, feature_set: str = None, custom_features: list = None,
                   validate_features: bool = True, generate_features: bool = True, scheduler_type: str = "none",
                   use_log_prices: bool = True, early_stopping_patience: int = 25, load_optuna_params: str = None,
                   load_optuna_dir: str = None, train_all_top: bool = False, seed: int = 42):
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        request_payload = {
            "action": "lstm_train",
            "rabbit_msg_id": rabbit_msg_id,
            "train_path": train_path,
            "lookback": lookback,
            "horizon": horizon,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "use_subset": use_subset,
            "subset_size": subset_size,
            "feature_set": feature_set,
            "custom_features": custom_features,
            "validate_features": validate_features,
            "generate_features": generate_features,
            "scheduler_type": scheduler_type,
            "use_log_prices": use_log_prices,
            "early_stopping_patience": early_stopping_patience,
            "load_optuna_params": load_optuna_params,
            "load_optuna_dir": load_optuna_dir,
            "train_all_top": train_all_top,
            "seed": seed
        }
        self.rabbitmq_service.publish(queue=ML_REQUEST_QUEUE, message=request_payload)
        print("Request for LSTM training sent to RabbitMQ")
        print(request_payload)
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )

    def lstm_test(self, model_dir: str = None, test_path: str = None, batch_size: int = 1024):
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        request_payload = {
            "action": "lstm_test",
            "rabbit_msg_id": rabbit_msg_id,
            "model_dir": model_dir,
            "test_path": test_path,
            "batch_size": batch_size
        }
        self.rabbitmq_service.publish(queue=ML_REQUEST_QUEUE, message=request_payload)
        print("Request for LSTM testing sent to RabbitMQ")
        print(request_payload)
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )

    def lstm_prepare_data(self, data_path: str, feature_set: str = "full", train_months: int = None,
                          agent_months: int = None, test_months: int = None, train_pct: float = 70.0,
                          agent_pct: float = 0.0, test_pct: float = 30.0, lookback: int = 48, horizon: int = 1,
                          scaler_features: str = "standard", scaler_price: str = "standard",
                          use_log_prices: bool = True):
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        request_payload = {
            "action": "lstm_prepare_data",
            "rabbit_msg_id": rabbit_msg_id,
            "data_path": data_path,
            "feature_set": feature_set,
            "train_months": train_months,
            "agent_months": agent_months,
            "test_months": test_months,
            "train_pct": train_pct,
            "agent_pct": agent_pct,
            "test_pct": test_pct,
            "lookback": lookback,
            "horizon": horizon,
            "scaler_features": scaler_features,
            "scaler_price": scaler_price,
            "use_log_prices": use_log_prices
        }
        self.rabbitmq_service.publish(queue=ML_REQUEST_QUEUE, message=request_payload)
        print("Request for LSTM data preparation sent to RabbitMQ")
        print(request_payload)
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )

    def lstm_tune(self, train_path: str, n_trials: int = 100, timeout: int = None, study_name: str = None,
                  subset_size: int = 200000, feature_set: str = None, custom_features: list = None,
                  validate_features: bool = True, generate_features: bool = True, fixed_horizon: int = None):
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        request_payload = {
            "action": "lstm_tune",
            "rabbit_msg_id": rabbit_msg_id,
            "train_path": train_path,
            "n_trials": n_trials,
            "timeout": timeout,
            "study_name": study_name,
            "subset_size": subset_size,
            "feature_set": feature_set,
            "custom_features": custom_features,
            "validate_features": validate_features,
            "generate_features": generate_features,
            "fixed_horizon": fixed_horizon
        }
        self.rabbitmq_service.publish(queue=ML_REQUEST_QUEUE, message=request_payload)
        print("Request for LSTM hyperparameter tuning sent to RabbitMQ")
        print(request_payload)
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )