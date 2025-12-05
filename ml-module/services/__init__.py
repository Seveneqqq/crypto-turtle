from services.rabbitmq_service import RabbitMQService
from services.rabbitmq_request_service import RabbitMQRequestService
from .sklearn_train_service import SklearnTrainServiceSimple
from .sklearn_data_prep_service import SklearnDataPrepService
from .sklearn_test_service import SklearnTestServiceSimple
from .sklearn_hyperparam_tuning_service import SklearnHyperparamTuningService
from .lstm_service import LSTMService
from configuration.rabbitmq_configuration import (
    RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USERNAME, RABBITMQ_PASSWORD
)

print("Initializing RabbitMQ Service...")
rabbitmq_service = RabbitMQService(
    host=RABBITMQ_HOST,
    port=RABBITMQ_PORT,
    username=RABBITMQ_USERNAME,
    password=RABBITMQ_PASSWORD
)

sklearn_train_service = SklearnTrainServiceSimple()
sklearn_dataprep_service = SklearnDataPrepService()
sklearn_test_service = SklearnTestServiceSimple()
sklearn_tuning_service = SklearnHyperparamTuningService()
lstm_service = LSTMService()

print("Initializing RabbitMQ Request Service...")
rabbitmq_request_service = RabbitMQRequestService(
    rabbitmq_service=rabbitmq_service,
    sklearn_train_service=sklearn_train_service,
    sklearn_dataprep_service=sklearn_dataprep_service,
    sklearn_test_service=sklearn_test_service,
    sklearn_tuning_service=sklearn_tuning_service,
    lstm_service=lstm_service
)

rabbitmq_service.request_service = rabbitmq_request_service
