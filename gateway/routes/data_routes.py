from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List

from services import auth_service, data_service
from pathlib import Path
import os

router = APIRouter(prefix="/data", tags=["data"])

# -------------------- STATIC PATHS --------------------
# foldery dokładnie takie, jakie podałeś
BASE_PATH = Path(os.getcwd()) / "shared/files"
DATA_PATHS = {
    "models": BASE_PATH / "models",
    "logs": BASE_PATH / "logs",
    "plots": BASE_PATH / "plots",
    "binance_raw": BASE_PATH / "binance_raw",
    "files": BASE_PATH,
    "agents_scikit": BASE_PATH / "agents/scikit",
    "raw_data": BASE_PATH / "raw_data",
    "processed_rlagent_data": BASE_PATH / "processed_rlagent_data",
    "processed_simulation_data": BASE_PATH / "processed_simulation_data",
    "processed_testmodel_data": BASE_PATH / "processed_testmodel_data",
    "processed_trainmodel_data": BASE_PATH / "processed_trainmodel_data",

    "processed_rlagent_data_lstm": "rlagent_lstm",
    "processed_simulation_data_lstm": "simulation_lstm",
    "processed_testmodel_data_lstm": "testmodel_lstm",
    "processed_trainmodel_data_lstm": "trainmodel_lstm",
    
    "models_ml": BASE_PATH / "models/ml",
    "models_sklearn": BASE_PATH / "models/sklearn",
}

# # upewnij się, że foldery istnieją
# for path in DATA_PATHS.values():
#     path.mkdir(parents=True, exist_ok=True)

# -------------------- ROUTES --------------------

@router.get("/private")
def private_data(user=Depends(auth_service.require_auth)):
    return {"msg": f"Witaj w data module, {user.user_metadata.get('name')}"}


# -------------------- MODELE --------------------

class ProcessDataRequest(BaseModel):
    filename: str
    steps: List[str] = ["load_data", "make_features", "add_technical_indicators"]
    preprocessing_method: str = "RAW"
    output_folders: List[str] = ["processed_trainmodel_data"]
    horizon: int = 1
    feature_sets: List[List[str]] = []

class TuneDataRequest(BaseModel):
    filename_candidates: List[str]
    steps: List[str]
    n_trials: int = 20
    technical_sets: List[str]
    preprocessing_methods: List[str]
    output_folders: List[str]  # <- dowolne nazwy folderów
    metrics: List[str]
    top_k_combinations: int = 5
    max_features_in_combo: int = 6

class ProcessDataLSTMRequest(BaseModel):
    filename: str
    steps: List[str] = ["load_data", "make_features", "add_technical_indicators"]
    preprocessing_method: str = "RAW"
    output_folders: List[str] = ["processed_trainmodel_data_lstm"]
    technical_indicators_set: str = "set1"
    selected_features: List[str] = []


class LSTMTuneDataRequest(BaseModel):
    filename_candidates: List[str]
    steps: List[str]
    n_trials: int = 20
    technical_sets: List[str]
    preprocessing_methods: List[str]
    output_folders: List[str] = ["processed_trainmodel_data"]
    top_k_combinations: int = 5
    max_features_in_combo: int = 6


# -------------------- ROUTES --------------------

@router.post("/tune-data")
def tune_data_route(request: TuneDataRequest):
    # konwertuj foldery na pełne ścieżki
    folders_full_paths = [
        str(DATA_PATHS.get(folder, BASE_PATH / folder)) for folder in request.output_folders
    ]

    data_service.request_tune_data(
        filename_candidates=request.filename_candidates,
        steps=request.steps,
        n_trials=request.n_trials,
        technical_sets=request.technical_sets,
        preprocessing_methods=request.preprocessing_methods,
        output_folders=folders_full_paths,
        metrics=request.metrics,
        top_k_combinations=request.top_k_combinations,
        max_features_in_combo=request.max_features_in_combo
    )
    return {"msg": "Classic Optuna request added to queue."}


@router.post("/tune-data-lstm")
def tune_data_lstm_route(request: LSTMTuneDataRequest):
    folders_full_paths = [
        str(DATA_PATHS.get(folder, BASE_PATH / folder)) for folder in request.output_folders
    ]
    data_service.request_tune_data_lstm(
        filename_candidates=request.filename_candidates,
        steps=request.steps,
        n_trials=request.n_trials,
        technical_sets=request.technical_sets,
        preprocessing_methods=request.preprocessing_methods,
        output_folders=folders_full_paths,
        metrics=["mse"],
        top_k_combinations=request.top_k_combinations,
        max_features_in_combo=request.max_features_in_combo
    )
    return {"msg": "LSTM Optuna request added to queue."}


@router.post("/process-data")
def process_data_route(request: ProcessDataRequest):
    if not request.filename.strip():
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    folders_full_paths = [
        str(DATA_PATHS.get(folder, BASE_PATH / folder))
        for folder in request.output_folders
    ]

    data_service.process_data(
        filename=request.filename.strip(),
        steps=request.steps,
        preprocessing_method=request.preprocessing_method,
        output_folders=folders_full_paths,
        horizon=request.horizon,
        feature_sets=request.feature_sets
    )

    return {"msg": "process_data request added to queue."}



@router.post("/process-data-lstm")
def process_data_lstm_route(request: ProcessDataLSTMRequest):
    if not request.filename.strip():
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    folders_full_paths = [
        str(DATA_PATHS.get(folder, BASE_PATH / folder))
        for folder in request.output_folders
    ]

    data_service.process_data_lstm(
        filename=request.filename.strip(),
        steps=request.steps,
        preprocessing_method=request.preprocessing_method,
        output_folders=folders_full_paths,
        technical_indicators_set=request.technical_indicators_set,
        selected_features=request.selected_features
    )

    return {"msg": "process_data_lstm request added to queue."}

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