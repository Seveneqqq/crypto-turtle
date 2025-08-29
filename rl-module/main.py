from fastapi import FastAPI

from configuration.rabbitmq_queues import RL_REQUEST_QUEUE
from services import rabbitmq_service

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from rl-module"}

rabbitmq_service.consume(queue=RL_REQUEST_QUEUE, callback = rabbitmq_service.handle_request)

