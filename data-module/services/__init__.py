from services.binance_service import BinanceService
from services.data_service import DataService
from services.optuna_service import OptunaService
from services.data_lstm_prepare_service import DataLSTMPrepareService
from services.lstm_optuna_service import LSTMOptunaService
from services.realtime_data_service import RealtimeDataService
from services.rabbitmq_service import RabbitMQService
from services.rabbitmq_request_service import RabbitMQRequestService
from configuration.rabbitmq_configuration import (
    RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USERNAME, RABBITMQ_PASSWORD
)
from configuration.configuration import BINANCE_API_KEY, BINANCE_API_URL

from services.data_service import DataService

print("Initializing RabbitMQ Service...")
rabbitmq_service = RabbitMQService(
    host=RABBITMQ_HOST,
    port=RABBITMQ_PORT,
    username=RABBITMQ_USERNAME,
    password=RABBITMQ_PASSWORD
)


print("Initializing Binance Service...")
binance_service = BinanceService(api_key=BINANCE_API_KEY, api_url=BINANCE_API_URL)

print("Initializing Data Service...")
data_service = DataService()

print("Initializing Data LSTM Prepare Service...")
data_lstm_prepare_service = DataLSTMPrepareService()

print("Initializing Realtime Data Service...")
realtime_data_service = RealtimeDataService(
    rabbitmq_service=rabbitmq_service
)

print("Initializing Optuna Service...")
optuna_service = OptunaService()

print("Initializing LSTM Optuna Service...")
lstm_optuna_service = LSTMOptunaService()

print("Initializing RabbitMQ Request Service...")
rabbitmq_request_service = RabbitMQRequestService(
    rabbitmq_service=rabbitmq_service,
    binance_service=binance_service,
    data_service=data_service,
    optuna_service=optuna_service,
    data_lstm_prepare_service=data_lstm_prepare_service,
    lstm_optuna_service=lstm_optuna_service
)

rabbitmq_service.request_service = rabbitmq_request_service
