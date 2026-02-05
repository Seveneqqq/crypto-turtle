# from services.binance_service import BinanceService
import os
if os.environ.get("SKIP_INIT") == "1":
    pass
else:
    from services.rabbitmq_service import RabbitMQService
    from services.simulation_service import SimulationService
    from services.visualization_service import VisualizationService
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


    print("Initializing Data Quality Service...")
    data_quality_service = DataQualityService()

    print("Initializing RabbitMQ Request Service...")
    rabbitmq_request_service = RabbitMQRequestService(
        rabbitmq_service=rabbitmq_service,
        simulation_service=simulation_service,
        visualization_service=visualization_service,
        data_quality_service=data_quality_service
    )

    rabbitmq_service.request_service = rabbitmq_request_service