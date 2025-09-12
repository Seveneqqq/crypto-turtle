from fastapi import APIRouter, File, Request, UploadFile
from fastapi.params import Depends
from services import auth_service
from services import rabbitmq_service
from services import data_service
from configuration.rabbitmq_queues import DATA_REQUEST_QUEUE

router = APIRouter(prefix="/data", tags=["data"])

@router.get("/private")
def private_data(
    user = Depends(auth_service.require_auth)
    ):
    return {"msg": f"Witaj w data module, {user.user_metadata.get('name')}"}


#Tutaj powinno byc wywolanie serwisu i w serwisie opublikowanie wiadomosci
#Przenieść do serwisow i zrobic walidacje, sprawdzenie czy pola sie zgadzaja i czy nie sa puste, wtedy mozna przekazac rabbitem
# @router.get("/rabbitest")
# def rabbit_test():
#     rabbitmq_service.publish(DATA_REQUEST_QUEUE, {
#         "task": "normalize_data",
#         "file": "dataset1.csv"
#         })
#     return {"msg": "Test message sent to RabbitMQ"}

@router.get("/get-historical-data")
def get_historical_data(
    symbol: str,
    interval: str,
    start_dt: str,
    end_dt: str,
    filename: str
):
    data_service.request_historical_data(
        symbol=symbol,
        interval=interval,
        start_dt=start_dt,
        end_dt=end_dt,
        filename=filename
    )
    return {"msg": "Request added to queue."}

