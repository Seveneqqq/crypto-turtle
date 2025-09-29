from typing import Dict
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.params import Depends
from services import tests_service
from pydantic import BaseModel

router = APIRouter(prefix="/tests", tags=["tests"])

#endpointy ktore wywolaja odpowiednia metode z danego serwisu i przekaze wiadomosc do kolejki

class SimulationRequest(BaseModel):
    balance_start: float
    time_step: int
    data_files: Dict[str, str]
    models: Dict[str, str]
    threshold_buy: float
    threshold_sell: float


@router.post("/simulate")
def simulate(req: SimulationRequest):
    tests_service.simulate(
        balance_start=req.balance_start,
        time_step=req.time_step,
        data_files=req.data_files,      # tu zmieniłem na data_paths, bo tak jest w serwisie
        models=req.models,        # tak samo models_paths
        threshold_buy=req.threshold_buy,
        threshold_sell=req.threshold_sell
    )
    return {"msg": "Request added to queue."}


class AccuracyRequest(BaseModel):
    data_dir: str
    data_file: str
    model_dir: str
    model_file: str

@router.post("/calculate_accuracy")
def calculate_accuracy(req: AccuracyRequest):
    tests_service.calculate_accuracy(
        data_dir=req.data_dir,
        data_file=req.data_file,
        model_dir=req.model_dir,
        model_file=req.model_file
    )
    return {"msg": "Request added to queue."}