from configuration.rabbitmq_queues import TEST_RESPONSE_QUEUE
from configuration.rabbitmq_queues import XAI_REQUEST_QUEUE

from datetime import datetime
class RabbitMQRequestService:
    def __init__(self, rabbitmq_service, simulation_service, visualization_service, data_quality_service, realtime_generate_decision_service):

        self.rabbitmq_service = rabbitmq_service
        self.simulation_service = simulation_service
        self.visualization_service = visualization_service
        self.data_quality_service = data_quality_service
        self.realtime_generate_decision_service = realtime_generate_decision_service

    def handle_request(self, data):
        print(data)
        match data.get("action"):
            case "simulate":
                print("Getting simulate data")
                result = self.simulation_service.run_simulation(
                    rabbit_msg_id=data.get("rabbit_msg_id"),
                    balance_start=data.get("balance_start"),
                    add_fee=data.get("add_fee"),
                    sim_mode=data.get("sim_mode"),
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
                    model_paths=data.get("model_paths"),
                    agent_rl=data.get("agent_rl"),
                    plots=data.get("plots"),
                    logs=data.get("logs"),
                    results=data.get("results")

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
            case "visualize":
                print("Getting visualize data")
                result = self.visualization_service.visualize(
                    rabbit_msg_id=data.get("rabbit_msg_id"),
                    simulation_files=data.get("simulation_files")

                )
                if result.get("result") == "success":
                    print("Successfully retrieved visualize data")
                    self.rabbitmq_service.publish(TEST_RESPONSE_QUEUE, result)
                else:
                    print("Failed to retrieve visualize data")
                    self.rabbitmq_service.publish(TEST_RESPONSE_QUEUE, {
                        "result": "failed",
                        "message": "Failed to retrieve visualize data"
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


    def handle_realtime_request(self, data):
        print("Getting realtime request ...")

        decisions_result = self.realtime_generate_decision_service.live_decision(results=data.get("result"))

        if decisions_result and decisions_result.get("result") == "success":
            print("Successfully processed realtime data")

            self.rabbitmq_service.publish(TEST_RESPONSE_QUEUE, decisions_result)
            self.rabbitmq_service.publish(XAI_REQUEST_QUEUE, decisions_result)

        else:
            print("Failed to process realtime data")

            self.rabbitmq_service.publish(TEST_RESPONSE_QUEUE, {
                "result": "failed",
                "message": "Failed to process realtime data"
            })
            self.rabbitmq_service.publish(XAI_REQUEST_QUEUE, {
                "result": "failed",
                "message": "Failed to process realtime data"
            })


                #Tutaj bedzie publikowanie danych z decision engine wraz z danymi ktore przyszly do xai i do serwisu ktory bedzie wykonywal operacje na binance

