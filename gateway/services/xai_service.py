from configuration.rabbitmq_queues import XAI_REQUEST_QUEUE
import os

class XaiService:
    """
    XaiService w gatewayu – przesyła requesty do xai-module przez RabbitMQ
    i zapisuje stan w SQLite. Gateway NIE wykonuje wyjaśnień, tylko przekazuje
    wszystkie parametry do xai-module.
    """

    def __init__(self, sqlite_service, rabbitmq_service):
        self.sqlite_service = sqlite_service
        self.rabbitmq_service = rabbitmq_service

    def handle_response(self, message: dict):
        """
        Odbiera odpowiedź z RabbitMQ i aktualizuje stan w SQLite.
        """
        print("Received response from RabbitMQ:")
        print(message)

        rabbit_msg_id = message.get("rabbit_msg_id")
        rabbit_result = message.get("result")

        if rabbit_msg_id:
            self.sqlite_service.update_message(
                rabbit_msg_id=rabbit_msg_id,
                state=rabbit_result,
                response=str(message)
            )
            print(f"Response for {rabbit_msg_id} updated in SQLite")
        else:
            print("Warning: rabbit_msg_id missing in response!")

    def request_explain_model(
        self,
        model_folder: str,
        model_file: str,
        explainer_type: str,
        data_folder: str,
        data_file: str,
        instance: int = None,
        feature: str = None,
        explainer_to_test: str = None
    ):
        """
        Przygotowuje i wysyła request do kolejki XAI.
        data_folder + data_file wskazują plik CSV, który będzie użyty do wyjaśnienia.
        """
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()

        payload = {
            "action": "explain_model",
            "rabbit_msg_id": rabbit_msg_id,
            "model_folder": model_folder,
            "model_file": model_file,
            "explainer_type": explainer_type,
            "data_folder":data_folder,
            "data_file": data_file,         
            "instance": instance,
            "feature": feature,
            "explainer_to_test": explainer_to_test
        }

        print("Sending explain_model request to RabbitMQ:")
        print(payload)

        self.rabbitmq_service.publish(queue=XAI_REQUEST_QUEUE, message=payload)

        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(payload)
        )

        return {"msg": "Explain model request sent to queue.", "rabbit_msg_id": rabbit_msg_id}
