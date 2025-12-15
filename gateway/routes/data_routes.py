from fastapi import APIRouter, File, Request, UploadFile
from fastapi.params import Depends
from services import auth_service
from services import rabbitmq_service
from services import data_service
from pydantic import BaseModel
from typing import List, Literal,Tuple
from configuration.rabbitmq_queues import DATA_REQUEST_QUEUE
from fastapi import HTTPException


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

VALID_OUTPUT_FOLDERS = ["trainmodel", "testmodel", "rlagent", "simulation","trainmodel_lstm", "testmodel_lstm", "rlagent_lstm", "simulation_lstm"]

class ProcessDataRequest(BaseModel):
    filename: str
    steps: List[str] = ["load_data", "make_features", "add_technical_indicators"]
    preprocessing_method: str 
    output_folders: List[Literal["trainmodel", "testmodel", "rlagent", "simulation","trainmodel_lstm", "testmodel_lstm", "rlagent_lstm", "simulation_lstm"]] = ["trainmodel"]
    technical_indicators_set: str

class TuneDataRequest(BaseModel):
    filename_candidates: List[str]
    steps: List[str]
    n_trials: int = 20
    technical_sets: List[str]
    preprocessing_methods: List[str]
    output_folders: List[Literal["trainmodel", "testmodel", "rlagent", "simulation","trainmodel_lstm", "testmodel_lstm", "rlagent_lstm", "simulation_lstm"]]
    metrics: List[str]

@router.post("/tune-data")
def tune_data_route(request: TuneDataRequest):
    data_service.request_tune_data(
        filename_candidates=request.filename_candidates,
        steps=request.steps,
        n_trials=request.n_trials,
        technical_sets=request.technical_sets,
        preprocessing_methods=request.preprocessing_methods,
        output_folders=request.output_folders,
        metrics=request.metrics
    )
    return {"msg": "Request added to queue."}

class LSTMTuneDataRequest(BaseModel):
    filename_candidates: List[str]
    steps: List[str]                      # np. ["load_data"]
    n_trials: int = 20
    technical_sets: List[str] = ["set1", "set2", "set3", "set4"]
    preprocessing_methods: List[str] = ["RAW", "DIFF", "LOG"]
    target_col: str = "close"
    # zakresy parametrów danych
    seq_len_range: Tuple[int, int, int] = (32, 128, 32)   # min, max, step
    scaler_types: List[str] = ["standard", "minmax"]
    # zakresy parametrów modelu
    lstm_units_range: Tuple[int, int, int] = (32, 128, 32)
    lstm_layers_range: Tuple[int, int] = (1, 3)
    dropout_range: Tuple[float, float, float] = (0.0, 0.5, 0.1)
    lr_range: Tuple[float, float] = (1e-4, 1e-2)
    batch_sizes: List[int] = [32, 64]
    epochs: int = 5

@router.post("/tune-data-lstm")
def tune_data_lstm_route(request: LSTMTuneDataRequest):
    data_service.request_tune_data_lstm(
        filename_candidates=request.filename_candidates,
        steps=request.steps,
        n_trials=request.n_trials,
        technical_sets=request.technical_sets,
        preprocessing_methods=request.preprocessing_methods,
        output_folders=["trainmodel_lstm"],   # cokolwiek, bo metoda tego wymaga
        metrics=["mse"],                      # cokolwiek, bo metoda tego wymaga
    )
    return {"msg": "LSTM tune request added to queue."}

@router.post("/process-data")
def process_data_route(request: ProcessDataRequest):
    filename = request.filename.strip()
    steps = request.steps
    preprocessing_method = request.preprocessing_method   # singular ✅
    technical_indicators_set = request.technical_indicators_set
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
        output_folders=output_folders,
        technical_indicators_set=technical_indicators_set
    )

    return {
        "msg": f"Processing request for '{filename}' added to queue.",
        "filename": filename,
        "steps": steps,
        "preprocessing_method": preprocessing_method,
        "output_folders": output_folders,
        "technical_indicators_set": technical_indicators_set
    }

@router.post("/process-data-lstm")
def process_data_route(request: ProcessDataRequest):
    filename = request.filename.strip()
    steps = request.steps
    preprocessing_method = request.preprocessing_method   # singular ✅
    technical_indicators_set = request.technical_indicators_set
    output_folders = request.output_folders or ["trainmodel"]

    if not filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    invalid_folders = [f for f in output_folders if f not in VALID_OUTPUT_FOLDERS]
    if invalid_folders:
        raise HTTPException(status_code=400, detail=f"Invalid output folders: {invalid_folders}")

    data_service.process_data_lstm(
        filename=filename,
        steps=steps,
        preprocessing_method=preprocessing_method,   
        output_folders=output_folders,
        technical_indicators_set=technical_indicators_set
    )

    return {
        "msg": f"Processing request for '{filename}' added to queue.",
        "filename": filename,
        "steps": steps,
        "preprocessing_method": preprocessing_method,
        "output_folders": output_folders,
        "technical_indicators_set": technical_indicators_set
    }
