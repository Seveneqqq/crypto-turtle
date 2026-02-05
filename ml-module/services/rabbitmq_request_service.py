from configuration.rabbitmq_queues import ML_RESPONSE_QUEUE


class RabbitMQRequestService:
    def __init__(self,
                 rabbitmq_service,
                 sklearn_train_service,
                 sklearn_dataprep_service,
                 sklearn_test_service,
                 sklearn_tuning_service,
                 lstm_service):
        self.rabbitmq_service = rabbitmq_service
        self.sklearn_train_service = sklearn_train_service
        self.sklearn_dataprep_service = sklearn_dataprep_service
        self.sklearn_test_service = sklearn_test_service
        self.sklearn_tuning_service = sklearn_tuning_service
        self.lstm_service = lstm_service

    def handle_request(self, data):
        print("Getting request ...")
        print(data)

        action = data.get("action")
        rabbit_msg_id = data.get("rabbit_msg_id")

        match action:
            case "train_model":
                print("Training model")
                result = self.sklearn_train_service.train_model(
                    rabbit_msg_id=rabbit_msg_id,
                    df_filename=data.get("df_filename"),
                    clf_version=data.get("clf_version"),
                    feature_cols=data.get("feature_cols")
                )
                self.rabbitmq_service.publish(ML_RESPONSE_QUEUE, result)

            case "prepare_features":
                print("Preparing dataset (features) from files")
                result = self.sklearn_dataprep_service.prepare_features_from_files(
                    rabbit_msg_id=rabbit_msg_id,
                    train_file=data.get("train_file"),
                    test_file=data.get("test_file"),
                    fe_version=data.get("fe_version", "fe_v2"),
                    feature_cols=data.get("feature_cols")
                )
                self.rabbitmq_service.publish(ML_RESPONSE_QUEUE, result)

            case "test_model":
                print("Testing model")
                result = self.sklearn_test_service.test_model(
                    rabbit_msg_id=rabbit_msg_id,
                    model_filename=data.get("model_filename"),
                    test_filename=data.get("test_filename")
                )
                self.rabbitmq_service.publish(ML_RESPONSE_QUEUE, result)

            case "tune_model":
                print("Tuning hyperparameters...")
                result = self.sklearn_tuning_service.tune_random_forest(
                    df_train_filename=data.get("df_train_filename"),
                    df_test_filename=data.get("df_test_filename"),
                    n_iter=int(data.get("n_iter", 50)),
                    cv_folds=int(data.get("cv_folds", 3)),
                    random_seed=int(data.get("random_seed", 42))
                )
                result["rabbit_msg_id"] = rabbit_msg_id
                self.rabbitmq_service.publish(ML_RESPONSE_QUEUE, result)

            case "lstm_train":
                print("Training LSTM model")
                result = self.lstm_service.train_model(
                    rabbit_msg_id=rabbit_msg_id,
                    train_path=data.get("train_path"),
                    lookback=data.get("lookback", 48),
                    horizon=data.get("horizon", 1),
                    hidden_size=data.get("hidden_size", 128),
                    num_layers=data.get("num_layers", 1),
                    dropout=data.get("dropout", 0.2),
                    epochs=data.get("epochs", 100),
                    batch_size=data.get("batch_size", 256),
                    lr=data.get("lr", 0.0005),
                    weight_decay=data.get("weight_decay", 1e-5),
                    use_subset=data.get("use_subset", False),
                    subset_size=data.get("subset_size", 100000),
                    feature_set=data.get("feature_set"),
                    custom_features=data.get("custom_features"),
                    validate_features=data.get("validate_features", True),
                    generate_features=data.get("generate_features", True),
                    scheduler_type=data.get("scheduler_type", "none"),
                    use_log_prices=data.get("use_log_prices", True),
                    early_stopping_patience=data.get("early_stopping_patience", 25),
                    load_optuna_params=data.get("load_optuna_params"),
                    load_optuna_dir=data.get("load_optuna_dir"),
                    train_all_top=data.get("train_all_top", False),
                    seed=data.get("seed", 42)
                )
                self.rabbitmq_service.publish(ML_RESPONSE_QUEUE, result)

            case "lstm_test":
                print("Testing LSTM model")
                result = self.lstm_service.test_model(
                    rabbit_msg_id=rabbit_msg_id,
                    model_dir=data.get("model_dir"),
                    test_path=data.get("test_path"),
                    batch_size=data.get("batch_size", 1024)
                )
                self.rabbitmq_service.publish(ML_RESPONSE_QUEUE, result)

            case "lstm_prepare_data":
                print("Preparing LSTM data")
                result = self.lstm_service.prepare_data(
                    rabbit_msg_id=rabbit_msg_id,
                    data_path=data.get("data_path"),
                    feature_set=data.get("feature_set", "full"),
                    train_months=data.get("train_months"),
                    agent_months=data.get("agent_months"),
                    test_months=data.get("test_months"),
                    train_pct=data.get("train_pct", 70.0),
                    agent_pct=data.get("agent_pct", 0.0),
                    test_pct=data.get("test_pct", 30.0),
                    lookback=data.get("lookback", 48),
                    horizon=data.get("horizon", 1),
                    scaler_features=data.get("scaler_features", "standard"),
                    scaler_price=data.get("scaler_price", "standard"),
                    use_log_prices=data.get("use_log_prices", True)
                )
                self.rabbitmq_service.publish(ML_RESPONSE_QUEUE, result)

            case "lstm_tune":
                print("Tuning LSTM hyperparameters")
                result = self.lstm_service.tune_hyperparameters(
                    rabbit_msg_id=rabbit_msg_id,
                    train_path=data.get("train_path"),
                    n_trials=data.get("n_trials", 100),
                    timeout=data.get("timeout"),
                    study_name=data.get("study_name"),
                    subset_size=data.get("subset_size", 200000),
                    feature_set=data.get("feature_set"),
                    custom_features=data.get("custom_features"),
                    validate_features=data.get("validate_features", True),
                    generate_features=data.get("generate_features", True),
                    fixed_horizon=data.get("fixed_horizon")
                )
                self.rabbitmq_service.publish(ML_RESPONSE_QUEUE, result)

            case _:
                print(f"[WARNING] Unknown action: {action}")
                self.rabbitmq_service.publish(ML_RESPONSE_QUEUE, {
                    "rabbit_msg_id": rabbit_msg_id,
                    "result": "failed",
                    "message": f"Unknown action: {action}"
                })


