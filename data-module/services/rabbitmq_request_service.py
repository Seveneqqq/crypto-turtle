from configuration.rabbitmq_queues import DATA_RESPONSE_QUEUE


class RabbitMQRequestService:
    def __init__(self, binance_service, rabbitmq_service):
        self.binance_service = binance_service
        self.rabbitmq_service = rabbitmq_service

    def handle_request(self, data):
        print(data)
        match data.get("action"):
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
                        "result": "error",
                        "message": "Failed to retrieve historical data"
                    })
