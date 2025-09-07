from fastapi import APIRouter, File, Request, UploadFile
from fastapi.params import Depends
from services import rl_service

router = APIRouter(prefix="/rl", tags=["rl"])

#endpointy ktore wywolaja odpowiednia metode z danego serwisu i przekaze wiadomosc do kolejki

@router.post("/simulate")
def simulate(
    balance_start: str,
    time_step: int,
    data_paths: str,
    models_paths: str,
):
    rl_service.simulate(
        balance_start=balance_start,
        time_step=time_step,
        data_paths=data_paths,
        models_paths=models_paths
    )
    return {"msg": "Request added to queue."}