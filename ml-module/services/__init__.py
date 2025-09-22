# old
# # from services.binance_service import BinanceService
# from services.rabbitmq_service import RabbitMQService
# from services.rabbitmq_request_service import RabbitMQRequestService
# from .sklearn_train_service import SklearnTrainService
# from configuration.rabbitmq_configuration import (
#     RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USERNAME, RABBITMQ_PASSWORD
# )
# # from configuration.configuration import BINANCE_API_KEY, BINANCE_API_URL
#
# print("Initializing RabbitMQ Service...")
# rabbitmq_service = RabbitMQService(
#     host=RABBITMQ_HOST,
#     port=RABBITMQ_PORT,
#     username=RABBITMQ_USERNAME,
#     password=RABBITMQ_PASSWORD
# )
#
# # print("Initializing Binance Service...")
# # binance_service = BinanceService(api_key=BINANCE_API_KEY, api_url=BINANCE_API_URL)
#
# sklearn_train_service = SklearnTrainService()
#
# print("Initializing RabbitMQ Request Service...")
# rabbitmq_request_service = RabbitMQRequestService(
#     rabbitmq_service=rabbitmq_service,
#     sklearn_train_service = sklearn_train_service
# )
#
# rabbitmq_service.request_service = rabbitmq_request_service


from services.rabbitmq_service import RabbitMQService
from services.rabbitmq_request_service import RabbitMQRequestService
from .sklearn_train_service import SklearnTrainServiceSimple
from .sklearn_data_prep_service import SklearnDataPrepService
from .sklearn_test_service import SklearnTestServiceSimple
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

print("Initializing RabbitMQ Request Service...")
rabbitmq_request_service = RabbitMQRequestService(
    rabbitmq_service=rabbitmq_service,
    sklearn_train_service=sklearn_train_service,
    sklearn_dataprep_service=sklearn_dataprep_service,
    sklearn_test_service=sklearn_test_service
)

rabbitmq_service.request_service = rabbitmq_request_service
