from typing import Dict
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.params import Depends
from services import rl_service
from pydantic import BaseModel

router = APIRouter(prefix="/rl", tags=["rl"])

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
    rl_service.simulate(
        balance_start=req.balance_start,
        time_step=req.time_step,
        data_paths=req.data_files,      # tu zmieniłem na data_paths, bo tak jest w serwisie
        models_paths=req.models,        # tak samo models_paths
        threshold_buy=req.threshold_buy,
        threshold_sell=req.threshold_sell
    )
    return {"msg": "Request added to queue."}