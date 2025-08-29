from fastapi import FastAPI

from configuration.rabbitmq_queues import ML_REQUEST_QUEUE
from services import rabbitmq_service

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from ml-module"}

rabbitmq_service.consume(queue=ML_REQUEST_QUEUE, callback = rabbitmq_service.handle_request)

