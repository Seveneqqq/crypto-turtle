from configuration.rabbitmq_queues import RL_REQUEST_QUEUE

class RlService:

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

    def train_agent_ppo(
            self,
            model_file,
            data_file,
            total_timesteps,
            learning_rate,
            n_steps,
            batch_size,
            gamma,
            ent_coef
    ):
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        request_payload = {
            "action": "train_agent_ppo",
            "rabbit_msg_id": rabbit_msg_id,  # Poprawic pola na podstawie configa od Rafała
            "model_file": model_file,
            "data_file": data_file,
            "total_timesteps": total_timesteps,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "gamma": gamma,
            "ent_coef": ent_coef,
        }
        print(request_payload)
        self.rabbitmq_service.publish(queue=RL_REQUEST_QUEUE, message=request_payload)
        print("Request for trainagentppo data sent to RabbitMQ")
        print(request_payload)
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )

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

        self.rabbitmq_service.publish(queue=RL_REQUEST_QUEUE, message=request_payload)
        print("Request with anything sent to RabbitMQ")
        print(request_payload)
        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(request_payload)
        )

