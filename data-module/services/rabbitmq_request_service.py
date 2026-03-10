from configuration.rabbitmq_queues import DATA_RESPONSE_QUEUE


class RabbitMQRequestService:
    def __init__(self, binance_service, rabbitmq_service, data_service,data_service_plus, optuna_service, data_lstm_prepare_service, lstm_optuna_service):
        self.binance_service = binance_service
        self.rabbitmq_service = rabbitmq_service
        self.data_service = data_service
        self.data_service_plus = data_service_plus
        self.optuna_service = optuna_service
        self.lstm_optuna_service = lstm_optuna_service
        self.data_lstm_prepare_service = data_lstm_prepare_service

    def handle_request(self, data):
        print(data)
        match data.get("action"):

            case "tune_data":
                print("Tuning data")
                result = self.optuna_service.run_optimization(data)
                if result.get("result") == "success":
                    print("Successfully tuned data")
                    self.rabbitmq_service.publish(DATA_RESPONSE_QUEUE, result)
                else:
                    print("Failed to tune data")
                    self.rabbitmq_service.publish(DATA_RESPONSE_QUEUE, {
                        "result": "failed",
                        "message": "Failed to tune data"
                    })

            case "tune_lstm":
                print("Tuning LSTM model & data")
                print(data)
                result = self.lstm_optuna_service.run_optimization(data)
            
                if result.get("result") == "success":
                    print("Successfully tuned LSTM")
                    self.rabbitmq_service.publish(DATA_RESPONSE_QUEUE, result)
                else:
                    print("Failed to tune LSTM")
                    self.rabbitmq_service.publish(DATA_RESPONSE_QUEUE, {
                        "rabbit_msg_id": data.get("rabbit_msg_id"),
                        "result": "failed",
                        "message": "Failed to tune LSTM"
                    })

            case "get_historical_data":
                print("Getting historical data")
                result = self.binance_service.get_historical_klines(
                    rabbitmq_msg_id=data.get("rabbit_msg_id"),
                    symbol=data.get("symbol"),
                    interval=data.get("interval"),
                    start_dt=data.get("start_dt"),
                    end_dt=data.get("end_dt"),
                    filename=data.get("filename")
                )
                if result.get("result") == "success":
                    print("Successfully retrieved historical data")
                    self.rabbitmq_service.publish(DATA_RESPONSE_QUEUE, result)
                else:
                    print("Failed to retrieve historical data")
                    self.rabbitmq_service.publish(DATA_RESPONSE_QUEUE, {
                        "result": "Failed",
                        "message": "Failed to retrieve historical data"
                    })
            
            case "process_data":
                print("Processing data")
            
                ds = self.data_service
                results = []
            
                filename = data.get("filename")
                horizon = data.get("horizon", 1)
                steps = data.get("steps", [])
                preprocessing = data.get("preprocessing_method", "RAW")
                output_folders = data.get("output_folders", [])
            
                feature_sets = data.get("feature_sets")
                if not feature_sets:
                    feature_sets = [ds.get_model_features()]
            
                try:
                    for output_folder in output_folders:
                        for feature_combo in feature_sets:
                            out_path = ds.generate_and_save(
                                filename=filename,
                                horizon=horizon,
                                output_folder=output_folder,
                                steps=steps,
                                preprocessing=preprocessing,
                                feature_combo=feature_combo,
                            )
                            results.append(out_path)
            
                    self.rabbitmq_service.publish(DATA_RESPONSE_QUEUE, {
                        "rabbit_msg_id": data.get("rabbit_msg_id"),
                        "result": "success",
                        "files": results
                    })
            
                except Exception as e:
                    self.rabbitmq_service.publish(DATA_RESPONSE_QUEUE, {
                        "rabbit_msg_id": data.get("rabbit_msg_id"),
                        "result": "failed",
                        "message": str(e)
                    })


            case "process_data_lstm":
                print("Processing data")
                ds = self.data_lstm_prepare_service
                ds.set_filename(data.get("filename"))

                try:
                    processed_info = ds.run_pipeline(
                        steps=data.get("steps", []),
                        preprocessing_method=data.get("preprocessing_method", "RAW"),
                        output_folders=data.get("output_folders", []),
                        technical_indicators_set=data.get("technical_indicators_set"),
                        selected_features=data.get("selected_features", []),
                    )

                    self.rabbitmq_service.publish(DATA_RESPONSE_QUEUE, {
                        "rabbit_msg_id": data.get("rabbit_msg_id"),
                        "result": "success",
                        "files": processed_info
                    })

                except Exception as e:
                    import traceback
                    error_text = traceback.format_exc()
                    print(error_text)

                    self.rabbitmq_service.publish(DATA_RESPONSE_QUEUE, {
                        "rabbit_msg_id": data.get("rabbit_msg_id"),
                        "result": "failed",
                        "message": str(e),
                        "trace": error_text
                    })