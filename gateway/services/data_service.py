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
        print(request_payload)
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )
