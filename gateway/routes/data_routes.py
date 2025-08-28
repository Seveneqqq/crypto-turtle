from fastapi import APIRouter, File, Request, UploadFile
from fastapi.params import Depends
from services import auth_service
from services import rabbitmq_service
from configuration.rabbitmq_queues import DATA_REQUEST_QUEUE

router = APIRouter(prefix="/data", tags=["data"])

@router.get("/private")
def private_data(
    user = Depends(auth_service.require_auth)
    ):
    return {"msg": f"Witaj w data, {user.user_metadata.get('name')}"}

@router.get("/rabbitest")
def rabbit_test():
    rabbitmq_service.publish(DATA_REQUEST_QUEUE, {
        "task": "test rabbitmq",
        "file": "dataset1.csv"
        })
    return {"msg": "Test message sent to RabbitMQ"}