from configuration.rabbitmq_queues import RL_RESPONSE_QUEUE

class RabbitMQRequestService:
    def __init__(self, rabbitmq_service, train_agent_ppo_service):
        
        self.rabbitmq_service = rabbitmq_service
        self.train_agent_ppo_service = train_agent_ppo_service

    def handle_request(self, data):
        print("Getting request ...")
        print(data)
        match data.get("action"):
            case "train_agent_ppo":
                print("Getting simulate data")
                result = self.train_agent_ppo_service.train_agent_ppo(
                    rabbitmq_msg_id=data.get("rabbit_msg_id"),
                    data_file= data.get("data_file"),
                    model_file=data.get("model_file"),
                    total_timesteps = data.get("total_timesteps"),
                    learning_rate = data.get("learning_rate"),
                    n_steps = data.get("n_steps"),
                    batch_size=data.get("batch_size"),
                    gamma = data.get("gamma"),
                    ent_coef = data.get("ent_coef"),

                )
                if result.get("result") == "success":
                    print("Successfully retrieved train_agent_ppo data")
                    self.rabbitmq_service.publish(RL_RESPONSE_QUEUE, result)
                else:
                    print("Failed to retrieve train_agent_ppo data")
                    self.rabbitmq_service.publish(RL_RESPONSE_QUEUE, {
                        "result": "failed",
                        "message": "Failed to retrieve train_agent_ppo data"
                    })
