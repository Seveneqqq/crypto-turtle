from configuration.rabbitmq_queues import DATA_RESPONSE_QUEUE


class RabbitMQRequestService:
    def __init__(self, binance_service, rabbitmq_service, data_service, optuna_service, data_lstm_prepare_service):
        self.binance_service = binance_service
        self.rabbitmq_service = rabbitmq_service
        self.data_service = data_service
        self.optuna_service = optuna_service
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
                ds.set_filename(data.get("filename"))
                processed_info = ds.run_pipeline(
                    steps=data.get("steps", []),
                    preprocessing_method=data.get("preprocessing_method", "RAW"),
                    output_folders=data.get("output_folders", None),
                    technical_indicators_set=data.get("technical_indicators_set")
                )

                if processed_info:
                    print("Successfully processed data")
                    self.rabbitmq_service.publish(DATA_RESPONSE_QUEUE, {
                        "rabbit_msg_id": data.get("rabbit_msg_id"),
                        "result": "success",
                        "files": processed_info
                    })
                else:
                    print("Failed to process data")
                    self.rabbitmq_service.publish(DATA_RESPONSE_QUEUE, {
                        "rabbit_msg_id": data.get("rabbit_msg_id"),
                        "result": "failed",
                        "message": "Failed to process data"
                    })

            case "process_data_lstm":
                print("Processing data")
                ds = self.data_lstm_prepare_service
                ds.set_filename(data.get("filename"))
            
                try:
                    processed_info = ds.run_pipeline(
                        steps=data.get("steps", []),
                        preprocessing_method=data.get("preprocessing_method", "RAW"),
                        output_folders=data.get("output_folders", None),
                        technical_indicators_set=data.get("technical_indicators_set")
                    )
            
                    print("Successfully processed data")
                    self.rabbitmq_service.publish(DATA_RESPONSE_QUEUE, {
                        "rabbit_msg_id": data.get("rabbit_msg_id"),
                        "result": "success",
                        "files": processed_info
                    })
            
                except Exception as e:
                    import traceback
                    error_text = traceback.format_exc()
                    print("\n=== ERROR DURING LSTM PROCESSING ===")
                    print(error_text)
                    print("===================================\n")
            
                    self.rabbitmq_service.publish(DATA_RESPONSE_QUEUE, {
                        "rabbit_msg_id": data.get("rabbit_msg_id"),
                        "result": "failed",
                        "message": str(e),
                        "trace": error_text
                    })

