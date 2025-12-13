from configuration.rabbitmq_queues import XAI_RESPONSE_QUEUE

class RabbitMQRequestService:
    def __init__(self, rabbitmq_service, xai_service):
        self.rabbitmq_service = rabbitmq_service
        self.xai_service = xai_service

    def handle_request(self, data: dict):
        print("Received RabbitMQ request:")
        print(data)

        match data.get("action"):
            case "explain_model":
                print("Explain model request received:")
                print(data)
                print("Starting model explanation...")
                
                result = self.xai_service.explain_model(
                    rabbit_msg_id=data.get("rabbit_msg_id"),
                    model_folder=data.get("model_folder"),
                    model_file=data.get("model_file"),
                    explainer_type=data.get("explainer_type"),
                    data_folder=data.get("data_folder"),
                    data_file=data.get("data_file"),
                    instance=data.get("instance"),
                    feature=data.get("feature"),
                    explainer_to_test=data.get("explainer_to_test")
                )

                if result.get("result") == "success":  # jeśli wysłano request poprawnie
                    print("Successfully sent explain_model request")
                    self.rabbitmq_service.publish(XAI_RESPONSE_QUEUE, {
                        "result": "success",
                        "rabbit_msg_id": result["rabbit_msg_id"]
                    })
                else:
                    print("Failed to send explain_model request")
                    print(result.get("message"))
                    print(result.get("result"))
                    self.rabbitmq_service.publish(XAI_RESPONSE_QUEUE, {
                        "result": "failed",
                        "message": "Failed to send explain_model request"
                    })
