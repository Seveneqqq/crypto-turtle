from configuration.rabbitmq_queues import TEST_REQUEST_QUEUE

class TestsService:

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
        add_fee,
        rlagent,
        rlagent_deterministic,
        slippage,
        wallet_percentage_limit,
        max_wallet_percentage_crypto,
        stop_loss,
        threshold_buy,
        threshold_sell,
        mincost,
        tx_fraction,
        max_fraction_when_confident,
        data_files,
        models,
        agents,
    ):
        
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        request_payload = {
            "action": "simulate",
            "rabbit_msg_id": rabbit_msg_id,  # Poprawic pola na podstawie configa od Rafała
            "balance_start": balance_start,
            "add_fee": add_fee,
            "rlagent": rlagent,
            "rlagent_deterministic": rlagent_deterministic,
            "slippage": slippage,
            "wallet_percentage_limit": wallet_percentage_limit,
            "max_wallet_percentage_crypto": max_wallet_percentage_crypto,
            "stop_loss": stop_loss,
            "threshold_buy": threshold_buy,
            "threshold_sell": threshold_sell,
            "mincost": mincost,
            "tx_fraction": tx_fraction,
            "max_fraction_when_confident": max_fraction_when_confident,
            "data_files": data_files,
            "models": models,
            "agents": agents,

        }
        print(request_payload)
        self.rabbitmq_service.publish(queue=TEST_REQUEST_QUEUE, message=request_payload)
        print("Request for simulation data sent to RabbitMQ")
        print(request_payload)
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )

