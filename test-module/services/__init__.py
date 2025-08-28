
from services.rabbitmq_service import RabbitMQService
from configuration.rabbitmq_configuration import (
    RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USERNAME, RABBITMQ_PASSWORD
)
from configuration.rabbitmq_queues import (
    DATA_REQUEST_QUEUE, DATA_RESPONSE_QUEUE,
    ML_REQUEST_QUEUE, ML_RESPONSE_QUEUE,
    RL_REQUEST_QUEUE, RL_RESPONSE_QUEUE,
    TEST_REQUEST_QUEUE, TEST_RESPONSE_QUEUE,
    XAI_REQUEST_QUEUE, XAI_RESPONSE_QUEUE
)

rabbitmq_service = RabbitMQService(
    host=RABBITMQ_HOST,
    port=RABBITMQ_PORT,
    username=RABBITMQ_USERNAME,
    password=RABBITMQ_PASSWORD
)


rabbitmq_service = RabbitMQService( host=RABBITMQ_HOST, port=RABBITMQ_PORT, username=RABBITMQ_USERNAME, password=RABBITMQ_PASSWORD )
rabbitmq_service.consume(queue=DATA_REQUEST_QUEUE, callback = data_service.handle_response)


