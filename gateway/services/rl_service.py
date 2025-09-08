from configuration.rabbitmq_queues import RL_REQUEST_QUEUE

class RlService:

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

    def simulate(
        self, 
        balance_start,
        time_step,
        data_files,
        models,
        threshold_buy,
        threshold_sell
    ):
        
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        request_payload = {
            "action": "simulate",
            "rabbit_msg_id": rabbit_msg_id,  # Poprawic pola na podstawie configa od Rafała
            "balance_start": balance_start,
            "time_step": time_step,
            "data_files": data_files,
            "models": models,
            "threshold_buy": threshold_buy,
            "threshold_sell": threshold_sell,
        }
        print(request_payload)
        self.rabbitmq_service.publish(queue=RL_REQUEST_QUEUE, message=request_payload)
        print("Request for simulation data sent to RabbitMQ")
        print(request_payload)
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )

