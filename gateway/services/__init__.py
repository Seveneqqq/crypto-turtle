from services.data_service import DataService
from services.file_service import FileService
from services.tests_service import TestsService
from services.ml_service import MlService
from .auth_service import AuthService
from configuration.configuration import SUPABASE_URL, SUPABASE_KEY, GOOGLE_REDIRECT_URI
from services.rabbitmq_service import RabbitMQService
from .sqlite_service import SQLiteService
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

# --- Inicjalizacja serwisów ---
auth_service = AuthService(SUPABASE_URL, SUPABASE_KEY, GOOGLE_REDIRECT_URI)
file_service = FileService()
sqlite_service = SQLiteService() 

# RabbitMQ service
rabbitmq_service = RabbitMQService(
    host=RABBITMQ_HOST,
    port=RABBITMQ_PORT,
    username=RABBITMQ_USERNAME,
    password=RABBITMQ_PASSWORD
)

data_service = DataService(sqlite_service, rabbitmq_service)
tests_service = TestsService(sqlite_service, rabbitmq_service)
ml_service = MlService(sqlite_service, rabbitmq_service)

# Konsumowanie wiadomości
rabbitmq_service.consume(queue=DATA_RESPONSE_QUEUE, callback=data_service.handle_response)
rabbitmq_service.consume(queue=ML_RESPONSE_QUEUE, callback=ml_service.handle_response)
rabbitmq_service.consume(queue=TEST_RESPONSE_QUEUE, callback=tests_service.handle_response)
# rabbitmq_service.consume(queue=TEST_RESPONSE_QUEUE)
# rabbitmq_service.consume(queue=XAI_RESPONSE_QUEUE)
