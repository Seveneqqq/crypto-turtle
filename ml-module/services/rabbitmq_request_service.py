from configuration.rabbitmq_queues import ML_RESPONSE_QUEUE


class RabbitMQRequestService:
    def __init__(self, rabbitmq_service, sklearn_train_service):
        
        self.rabbitmq_service = rabbitmq_service
        self.sklearn_train_service = sklearn_train_service

    def handle_request(self, data):
        print(data)
        match data.get("action"):
            case "train_model":
                print("Training model")
                result = self.sklearn_train_service.train_model(
                    rabbit_msg_id=data.get("rabbit_msg_id"),
                    df_filename=data.get("df_filename"), 
                    fe_version=data.get("fe_version"),
                    target_version=data.get("target_version"),
                    clf_version=data.get("clf_version"),
                )
                if result.get("result") == "success":
                    print("Successfully trained model")
                    self.rabbitmq_service.publish(ML_RESPONSE_QUEUE, result)
                else:
                    print("Failed to failed to train model")
                    self.rabbitmq_service.publish(ML_RESPONSE_QUEUE, {
                        "result": "error",
                        "message": "Failed to failed to train model"
                    })
