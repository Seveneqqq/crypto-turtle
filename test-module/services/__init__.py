# from services.binance_service import BinanceService
from services.rabbitmq_service import RabbitMQService
from services.simulation_service import SimulationService
from services.visualization_service import VisualizationService
from services.accuracy_service import AccuracyService
from services.data_quality_service import DataQualityService
from services.rabbitmq_request_service import RabbitMQRequestService
from configuration.rabbitmq_configuration import (
    RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USERNAME, RABBITMQ_PASSWORD
)
# from configuration.configuration import BINANCE_API_KEY, BINANCE_API_URL

print("Initializing RabbitMQ Service...")
rabbitmq_service = RabbitMQService(
    host=RABBITMQ_HOST,
    port=RABBITMQ_PORT,
    username=RABBITMQ_USERNAME,
    password=RABBITMQ_PASSWORD
)

print("Initializing Simulation Service...")
simulation_service = SimulationService()

print("Initializing Visualization Service...")
visualization_service = VisualizationService()

print("Initializing Accuracy Service...")
accuracy_service = AccuracyService()
print("Initializing Data Quality Service...")
data_quality_service = DataQualityService()

print("Initializing RabbitMQ Request Service...")
rabbitmq_request_service = RabbitMQRequestService(
    rabbitmq_service=rabbitmq_service,
    simulation_service=simulation_service,
    visualization_service=visualization_service,
    accuracy_service=accuracy_service
    data_quality_service=data_quality_service
)

rabbitmq_service.request_service = rabbitmq_request_service