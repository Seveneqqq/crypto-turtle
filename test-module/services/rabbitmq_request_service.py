from configuration.rabbitmq_queues import TEST_RESPONSE_QUEUE


class RabbitMQRequestService:
    def __init__(self, rabbitmq_service, simulation_service, data_quality_service):

        self.rabbitmq_service = rabbitmq_service
        self.simulation_service = simulation_service
        self.data_quality_service = data_quality_service

    def handle_request(self, data):
        print(data)
        match data.get("action"):
            case "simulate":
                print("Getting simulate data")
                result = self.simulation_service.run_simulation(
                    rabbit_msg_id=data.get("rabbit_msg_id"),
                    balance_start=data.get("balance_start"),
                    add_fee=data.get("add_fee"),
                    rlagent=data.get("rlagent"),
                    rlagent_deterministic=data.get("rlagent_deterministic"),
                    slippage=data.get("slippage"),
                    wallet_percentage_limit=data.get("wallet_percentage_limit"),
                    max_wallet_percentage_crypto=data.get("max_wallet_percentage_crypto"),
                    stop_loss=data.get("stop_loss"),
                    threshold_buy=data.get("threshold_buy"),
                    threshold_sell=data.get("threshold_sell"),
                    mincost=data.get("mincost"),
                    tx_fraction=data.get("tx_fraction"),
                    max_fraction_when_confident=data.get("max_fraction_when_confident"),
                    data_files=data.get("data_files"),
                    models=data.get("models"),
                    agents=data.get("agents"),

                )
                if result.get("result") == "success":
                    print("Successfully retrieved simulate data")
                    self.rabbitmq_service.publish(TEST_RESPONSE_QUEUE, result)
                else:
                    print("Failed to retrieve simulate data")
                    self.rabbitmq_service.publish(TEST_RESPONSE_QUEUE, {
                        "result": "failed",
                        "message": "Failed to retrieve simulation data"
                    })

            case "calculate_quality":
                print("Calculating data quality")
                result = self.data_quality_service.calculate_quality(
                    rabbit_msg_id=data.get("rabbit_msg_id"),
                    data_dir=data.get("data_dir"),
                    data_file=data.get("data_file"),
                )
                if result.get("result") == "success":
                    print("Successfully calculated data quality")
                    self.rabbitmq_service.publish(TEST_RESPONSE_QUEUE, result)
                else:
                    print("Failed to calculate data quality")
                    self.rabbitmq_service.publish(TEST_RESPONSE_QUEUE, result)

                    
