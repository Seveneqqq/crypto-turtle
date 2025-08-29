from fastapi import FastAPI

from configuration.rabbitmq_queues import XAI_REQUEST_QUEUE
from services import rabbitmq_service

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from xai-module"}

rabbitmq_service.consume(queue=XAI_REQUEST_QUEUE, callback = rabbitmq_service.handle_request)

