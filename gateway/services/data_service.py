from services import rabbitmq_service
from configuration.rabbitmq_queues import DATA_REQUEST_QUEUE

class DataService:
    
    def __init__(self):
        self.rabbitmq_service = rabbitmq_service

    def handle_response(self, message):
        print(f"Received response: {message}")