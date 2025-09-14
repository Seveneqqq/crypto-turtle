from configuration.rabbitmq_queues import ML_REQUEST_QUEUE

class MlService:

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

    def just_send_data_to_module(self,data):  
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        
        action = data.get("action")
        if not action:
            raise ValueError("Missing 'action' in request payload")

        request_payload = {
            "action": action,
            "rabbit_msg_id": rabbit_msg_id,
            "data":data
        }

        self.rabbitmq_service.publish(queue=ML_REQUEST_QUEUE, message=request_payload)
        print("Request with anything sent to RabbitMQ")
        print(request_payload)
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )


    def train_model(
        self,
        df_filename: str,
        fe_version: str,
        target_version: str,
        clf_version: str,
        # feature_cols: list,
        # target_col: str
    ):

        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()

        request_payload = {
            "action": "train_model",
            "rabbit_msg_id": rabbit_msg_id,
            "df_filename": df_filename,
            "fe_version": fe_version,
            "target_version": target_version,
            "clf_version": clf_version,
            # "feature_cols": feature_cols,
            # "target_col": target_col,
        }

        self.rabbitmq_service.publish(queue=ML_REQUEST_QUEUE, message=request_payload)
        print("Request for model training sent to RabbitMQ")
        print(request_payload)
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )

