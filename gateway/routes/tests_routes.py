from typing import Dict, Union, List
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.params import Depends
from services import tests_service
from pydantic import BaseModel

router = APIRouter(prefix="/tests", tags=["tests"])

#endpointy ktore wywolaja odpowiednia metode z danego serwisu i przekaze wiadomosc do kolejki

class SimulationRequest(BaseModel):
    balance_start: float
    add_fee: bool
    sim_mode: str
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
    model_paths: Dict[str, str]
    agents: Dict[str, str]
    plots: bool
    logs: bool
    results: bool


@router.post("/simulate")
def simulate(req: SimulationRequest):
    tests_service.simulate(
        balance_start=req.balance_start,
        add_fee=req.add_fee,
        sim_mode=req.sim_mode,
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
        model_paths=req.model_paths,
        agents=req.agents,
        plots=req.plots,
        logs=req.logs,
        results=req.results
    )
    return {"msg": "Request added to queue."}

class VisualizationRequest(BaseModel):
     simulation_files: Union[str, List[str]]

@router.post("/visualize")
def visualize(req: VisualizationRequest):
    tests_service.visualize(
        simulation_files=req.simulation_files,
    )
    return {"msg": "Request added to queue."}

class DataQualityRequest(BaseModel):
    data_dir: str
    data_file: str

@router.post("/calculate_quality")
def calculate_quality(req: DataQualityRequest):
    result = tests_service.calculate_quality(
        data_dir=req.data_dir,
        data_file=req.data_file
    )
    return {"msg": "Request processed.", "result": result}
