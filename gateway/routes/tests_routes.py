from typing import Dict
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.params import Depends
from services import tests_service
from pydantic import BaseModel

router = APIRouter(prefix="/tests", tags=["tests"])

#endpointy ktore wywolaja odpowiednia metode z danego serwisu i przekaze wiadomosc do kolejki

class SimulationRequest(BaseModel):
    balance_start: float
    add_fee: bool
    rlagent: bool
    rlagent_deterministic: bool
    slippage: int
    wallet_percentage_limit: float
    max_wallet_percentage_crypto: float
    stop_loss: float
    threshold_buy: float
    threshold_sell: float
    mincost: float
    tx_fraction: float
    max_fraction_when_confident: float
    data_files: Dict[str, str]
    models: Dict[str, str]
    agents: Dict[str, str]


@router.post("/simulate")
def simulate(req: SimulationRequest):
    tests_service.simulate(
        balance_start=req.balance_start,
        add_fee=req.add_fee,
        rlagent=req.rlagent,
        rlagent_deterministic=req.rlagent_deterministic,
        slippage=req.slippage,
        wallet_percentage_limit=req.wallet_percentage_limit,
        max_wallet_percentage_crypto=req.max_wallet_percentage_crypto,
        stop_loss=req.stop_loss,
        threshold_buy=req.threshold_buy,
        threshold_sell=req.threshold_sell,
        mincost=req.mincost,
        tx_fraction=req.tx_fraction,
        max_fraction_when_confident=req.max_fraction_when_confident,
        data_files=req.data_files,
        models=req.models,
        agents=req.agents,
    )
    return {"msg": "Simulation Request added to queue."}