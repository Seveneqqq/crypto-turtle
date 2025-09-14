from configuration.rabbitmq_queues import RL_RESPONSE_QUEUE

class RabbitMQRequestService:
    def __init__(self, rabbitmq_service, simulation_service):
        
        self.rabbitmq_service = rabbitmq_service
        self.simulation_service = simulation_service

    def handle_request(self, data):
        print("Getting request ...")
        print(data)
        match data.get("action"):
            case "simulate":
                print("Getting simulate data")
                result = self.simulation_service.run_simulation(
                    rabbitmq_msg_id=data.get("rabbit_msg_id"),
                    balance_start=data.get("balance_start"),
                    time_step=data.get("time_step"),
                    data_files=data.get("data_files"),
                    models=data.get("models"),
                    threshold_buy=data.get("threshold_buy"),
                    threshold_sell=data.get("threshold_sell")
                )
                if result.get("result") == "success":
                    print("Successfully retrieved simulate data")
                    self.rabbitmq_service.publish(RL_RESPONSE_QUEUE, result)
                else:
                    print("Failed to retrieve simulate data")
                    self.rabbitmq_service.publish(RL_RESPONSE_QUEUE, {
                        "result": "failed",
                        "message": "Failed to retrieve simulation data"
                    })
