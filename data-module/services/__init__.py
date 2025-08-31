from services.binance_service import BinanceService
from services.rabbitmq_service import RabbitMQService
from services.rabbitmq_request_service import RabbitMQRequestService
from configuration.rabbitmq_configuration import (
    RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USERNAME, RABBITMQ_PASSWORD
)
from configuration.configuration import BINANCE_API_KEY, BINANCE_API_URL

print("Initializing RabbitMQ Service...")
rabbitmq_service = RabbitMQService(
    host=RABBITMQ_HOST,
    port=RABBITMQ_PORT,
    username=RABBITMQ_USERNAME,
    password=RABBITMQ_PASSWORD
)

print("Initializing Binance Service...")
binance_service = BinanceService(api_key=BINANCE_API_KEY, api_url=BINANCE_API_URL)

print("Initializing RabbitMQ Request Service...")
rabbitmq_request_service = RabbitMQRequestService(
    rabbitmq_service=rabbitmq_service,
    binance_service=binance_service
)

rabbitmq_service.request_service = rabbitmq_request_service
