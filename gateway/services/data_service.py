from configuration.rabbitmq_queues import DATA_REQUEST_QUEUE

class DataService:

    def __init__(self, sqlite_service, rabbitmq_service):
        self.sqlite_service = sqlite_service
        self.rabbitmq_service = rabbitmq_service

    def handle_response(self, message):
        print("Received response from RabbitMQ")
        print(message)
        rabbit_msg_id = message.get("rabbit_msg_id")
        rabbit_result = message.get("result")
        self.sqlite_service.update_message(
            rabbit_msg_id=rabbit_msg_id,
            state=rabbit_result,
            response=str(message)
        )
        print("Response updated in SQLite")

    def request_tune_data(self, filename_candidates, steps, n_trials, technical_sets, preprocessing_methods, output_folders, metrics):
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        request_payload = {
            "action": "tune_data",
            "rabbit_msg_id": rabbit_msg_id,
            "filename_candidates": filename_candidates,
            "steps": steps,
            "n_trials": n_trials,
            "technical_sets": technical_sets,
            "preprocessing_methods": preprocessing_methods,
            "output_folders": output_folders,
            "metrics": metrics,
        }
        print(request_payload)
        self.rabbitmq_service.publish(queue=DATA_REQUEST_QUEUE, message=request_payload)
        print("Request for tune data sent to RabbitMQ")
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )

    def request_tune_data_lstm(
    self,
    filename_candidates,
    steps,
    n_trials,
    technical_sets,
    preprocessing_methods,
    output_folders,
    metrics,
):
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        request_payload = {
            "action": "tune_lstm",
            "rabbit_msg_id": rabbit_msg_id,
            "filename_candidates": filename_candidates,
            "steps": steps,
            "n_trials": n_trials,
            "technical_sets": technical_sets,
            "preprocessing_methods": preprocessing_methods,
            "output_folders": output_folders,
            "metrics": metrics,
        }
        print(request_payload)
        self.rabbitmq_service.publish(queue=DATA_REQUEST_QUEUE, message=request_payload)
        print("Request for tune LSTM data sent to RabbitMQ")
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )

    def request_historical_data(self, symbol, interval, start_dt, end_dt, filename):
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        request_payload = {
            "action": "get_historical_data",
            "rabbit_msg_id": rabbit_msg_id,
            "symbol": symbol,
            "interval": interval,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "filename": filename
        }
        print(request_payload)
        self.rabbitmq_service.publish(queue=DATA_REQUEST_QUEUE, message=request_payload)
        print("Request for historical data sent to RabbitMQ")
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )

    def process_data(self, filename, steps=None, preprocessing_method=None, output_folders=None, technical_indicators_set=None):
        steps = steps or ["load_data", "make_features", "add_technical_indicators"]
        preprocessing_method = preprocessing_method or "RAW"   # singular ✅
        output_folders = output_folders or ["trainmodel"]
        technical_indicators_set = technical_indicators_set or "default"

        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
    
        request_payload = {
            "action": "process_data",
            "rabbit_msg_id": rabbit_msg_id,
            "filename": filename,
            "steps": steps,
            "preprocessing_method": preprocessing_method,   # singular ✅
            "output_folders": output_folders,
            "technical_indicators_set": technical_indicators_set
        }
    
        print("Sending process_data request to RabbitMQ:")
        print(request_payload)
    
        self.rabbitmq_service.publish(queue=DATA_REQUEST_QUEUE, message=request_payload)
    
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )
    
    def process_data_lstm(self, filename, steps=None, preprocessing_method=None, output_folders=None, technical_indicators_set=None):
        steps = steps or ["load_data", "make_features", "add_technical_indicators"]
        preprocessing_method = preprocessing_method or "RAW"   # singular ✅
        output_folders = output_folders or ["trainmodel"]
        technical_indicators_set = technical_indicators_set or "default"

        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
    
        request_payload = {
            "action": "process_data_lstm",
            "rabbit_msg_id": rabbit_msg_id,
            "filename": filename,
            "steps": steps,
            "preprocessing_method": preprocessing_method,   # singular ✅
            "output_folders": output_folders,
            "technical_indicators_set": technical_indicators_set
        }
    
        print("Sending process_data_lstm request to RabbitMQ:")
        print(request_payload)
    
        self.rabbitmq_service.publish(queue=DATA_REQUEST_QUEUE, message=request_payload)
    
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )