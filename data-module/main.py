from fastapi import FastAPI

from configuration.rabbitmq_queues import DATA_REQUEST_QUEUE
from services import rabbitmq_service

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from data-module"}

rabbitmq_service.consume(queue=DATA_REQUEST_QUEUE, callback = rabbitmq_service.handle_response)

