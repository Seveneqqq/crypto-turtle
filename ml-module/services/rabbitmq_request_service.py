# old
# from configuration.rabbitmq_queues import ML_RESPONSE_QUEUE
#
#
# class RabbitMQRequestService:
#     def __init__(self, rabbitmq_service, sklearn_train_service):
#
#         self.rabbitmq_service = rabbitmq_service
#         self.sklearn_train_service = sklearn_train_service
#
#     def handle_request(self, data):
#         print("Getting request ...")
#         print(data)
#         match data.get("action"):
#             case "train_model":
#                 print("Training model")
#                 result = self.sklearn_train_service.train_model(
#                     rabbit_msg_id=data.get("rabbit_msg_id"),
#                     df_filename=data.get("df_filename"),
#                     fe_version=data.get("fe_version"),
#                     target_version=data.get("target_version"),
#                     clf_version=data.get("clf_version"),
#                 )
#                 if result.get("result") == "success":
#                     print("Successfully trained model")
#                     self.rabbitmq_service.publish(ML_RESPONSE_QUEUE, result)
#                 else:
#                     print("Failed to failed to train model")
#                     self.rabbitmq_service.publish(ML_RESPONSE_QUEUE, {
#                         "result": "failed",
#                         "message": "Failed to failed to train model"
#                     })

from configuration.rabbitmq_queues import ML_RESPONSE_QUEUE


class RabbitMQRequestService:
    def __init__(self, rabbitmq_service, sklearn_train_service, sklearn_dataprep_service, sklearn_test_service):
        self.rabbitmq_service = rabbitmq_service
        self.sklearn_train_service = sklearn_train_service
        self.sklearn_dataprep_service = sklearn_dataprep_service
        self.sklearn_test_service = sklearn_test_service

    def handle_request(self, data):
        print("Getting request ...")
        print(data)

        action = data.get("action")
        rabbit_msg_id = data.get("rabbit_msg_id")

        match action:
            case "train_model":
                print("Training model")
                result = self.sklearn_train_service.train_model(
                    rabbit_msg_id=rabbit_msg_id,
                    df_filename=data.get("df_filename"),
                    clf_version=data.get("clf_version"),
                    feature_cols=data.get("feature_cols")
                )
                self.rabbitmq_service.publish(ML_RESPONSE_QUEUE, result)

            case "prepare_features":
                print("Preparing dataset (features) from files")
                result = self.sklearn_dataprep_service.prepare_features_from_files(
                    rabbit_msg_id=rabbit_msg_id,
                    train_file=data.get("train_file"),
                    test_file=data.get("test_file"),
                    fe_version=data.get("fe_version", "fe_v2"),
                    feature_cols=data.get("feature_cols")
                )
                self.rabbitmq_service.publish(ML_RESPONSE_QUEUE, result)

            case "test_model":
                print("Testing model")
                result = self.sklearn_test_service.test_model(
                    rabbit_msg_id=rabbit_msg_id,
                    model_filename=data.get("model_filename"),
                    test_filename=data.get("test_filename")
                )
                self.rabbitmq_service.publish(ML_RESPONSE_QUEUE, result)

            case _:
                print(f"[WARNING] Unknown action: {action}")
                self.rabbitmq_service.publish(ML_RESPONSE_QUEUE, {
                    "rabbit_msg_id": rabbit_msg_id,
                    "result": "failed",
                    "message": f"Unknown action: {action}"
                })

