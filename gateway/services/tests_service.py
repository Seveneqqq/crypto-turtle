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
        sim_mode,
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
        model_paths,
        agent_rl,
        plots,
        logs,
        results
    ):
        
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        request_payload = {
            "action": "simulate",
            "rabbit_msg_id": rabbit_msg_id,  # Poprawic pola na podstawie configa od Rafała
            "balance_start": balance_start,
            "add_fee": add_fee,
            "sim_mode": sim_mode,
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
            "model_paths": model_paths,
            "agent_rl": agent_rl,
            "plots": plots,
            "logs": logs,
            "results": results

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

    def visualize(
            self,
            simulation_files
    ):

        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        request_payload = {
            "action": "visualize",
            "rabbit_msg_id": rabbit_msg_id,  # Poprawic pola na podstawie configa od Rafała
            "simulation_files": simulation_files

        }
        print(request_payload)
        self.rabbitmq_service.publish(queue=TEST_REQUEST_QUEUE, message=request_payload)
        print("Request for visualization data sent to RabbitMQ")
        print(request_payload)
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )

    def calculate_quality(self, data_dir, data_file):
        allowed_data_dirs = {
            "processed_trainmodel_data",
            "processed_testmodel_data",
            "processed_rlagent_data",
            "processed_simulation_data",
            "raw_data",
        }

        if data_dir not in allowed_data_dirs:
            print(f"Invalid data_dir: {data_dir}. Must be one of {allowed_data_dirs}")
            return

        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        request_payload = {
            "action": "calculate_quality",
            "rabbit_msg_id": rabbit_msg_id,
            "data_dir": data_dir,
            "data_file": data_file
        }
        print(request_payload)
        self.rabbitmq_service.publish(queue=TEST_REQUEST_QUEUE, message=request_payload)
        print("Request for calculate quality data sent to RabbitMQ")
        print(request_payload)
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )