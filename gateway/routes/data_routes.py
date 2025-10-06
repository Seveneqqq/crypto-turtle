from fastapi import APIRouter, File, Request, UploadFile
from fastapi.params import Depends
from services import auth_service
from services import rabbitmq_service
from services import data_service
from pydantic import BaseModel
from typing import List, Literal
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

VALID_OUTPUT_FOLDERS = ["trainmodel", "testmodel", "rlagent", "simulation"]

class ProcessDataRequest(BaseModel):
    filename: str
    steps: List[str] = ["load_data", "make_features", "add_technical_indicators"]
    preprocessing_method: str 
    output_folders: List[Literal["trainmodel", "testmodel", "rlagent", "simulation"]] = ["trainmodel"]

@router.post("/process-data")
def process_data_route(request: ProcessDataRequest):
    filename = request.filename.strip()
    steps = request.steps
    preprocessing_method = request.preprocessing_method   # singular ✅
    output_folders = request.output_folders or ["trainmodel"]

    if not filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    invalid_folders = [f for f in output_folders if f not in VALID_OUTPUT_FOLDERS]
    if invalid_folders:
        raise HTTPException(status_code=400, detail=f"Invalid output folders: {invalid_folders}")

    data_service.process_data(
        filename=filename,
        steps=steps,
        preprocessing_method=preprocessing_method,   
        output_folders=output_folders
    )

    return {
        "msg": f"Processing request for '{filename}' added to queue.",
        "filename": filename,
        "steps": steps,
        "preprocessing_method": preprocessing_method, 
        "output_folders": output_folders
    }
