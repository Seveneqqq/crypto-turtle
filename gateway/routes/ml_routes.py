from typing import Any, Dict, List
from fastapi import APIRouter, Body
from pydantic import BaseModel
from services import ml_service

router = APIRouter(prefix="/ml", tags=["ml"])

class TrainRequest(BaseModel):
    df_filename: str
    clf_version: str
    feature_cols: List[str] = None

class PrepareFeaturesRequest(BaseModel):
    train_file: str
    test_file: str
    fe_version: str = "fe_v2"
    feature_cols: List[str] = None

class TestModelRequest(BaseModel):
    model_filename: str
    test_filename: str

class TuneModelRequest(BaseModel):
    df_train_filename: str
    df_test_filename: str
    n_iter: int = 50
    cv_folds: int = 3
    random_seed: int = 42

class LSTMTrainRequest(BaseModel):
    train_path: str
    lookback: int = 48
    horizon: int = 1
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    epochs: int = 100
    batch_size: int = 256
    lr: float = 0.0005
    weight_decay: float = 1e-5
    use_subset: bool = False
    subset_size: int = 100000
    feature_set: str = None
    custom_features: List[str] = None
    validate_features: bool = True
    generate_features: bool = True
    scheduler_type: str = "none"
    use_log_prices: bool = True
    early_stopping_patience: int = 25
    load_optuna_params: str = None
    load_optuna_dir: str = None
    train_all_top: bool = False
    seed: int = 42

class LSTMTestRequest(BaseModel):
    model_dir: str = None
    test_path: str = None
    batch_size: int = 1024

class LSTMPrepareDataRequest(BaseModel):
    data_path: str
    feature_set: str = "full"
    train_months: int = None
    agent_months: int = None
    test_months: int = None
    train_pct: float = 70.0
    agent_pct: float = 0.0
    test_pct: float = 30.0
    lookback: int = 48
    horizon: int = 1
    scaler_features: str = "standard"
    scaler_price: str = "standard"
    use_log_prices: bool = True

class LSTMTuneRequest(BaseModel):
    train_path: str
    n_trials: int = 100
    timeout: int = None
    study_name: str = None
    subset_size: int = 200000
    feature_set: str = None
    custom_features: List[str] = None
    validate_features: bool = True
    generate_features: bool = True
    fixed_horizon: int = None

@router.post("/train")
async def train_model_endpoint(req: TrainRequest):
    ml_service.train_model(
        df_filename=req.df_filename,
        clf_version=req.clf_version,
        feature_cols=req.feature_cols
    )
    return {"msg": "Request added to queue."}

@router.post("/prepare_features")
async def prepare_features_endpoint(req: PrepareFeaturesRequest):
    ml_service.prepare_features(
        train_file=req.train_file,
        test_file=req.test_file,
        fe_version=req.fe_version,
        feature_cols=req.feature_cols
    )
    return {"msg": "Prepare features request added to queue."}

@router.post("/test_model")
async def test_model_endpoint(req: TestModelRequest):
    ml_service.test_model(
        model_filename=req.model_filename,
        test_filename=req.test_filename
    )
    return {"msg": "Test model request added to queue."}

@router.post("/anything")
async def just_send_data_to_module(req: Dict[str, Any] = Body(...)):
    ml_service.just_send_data_to_module(data=req)
    return {"status": "ok"}

@router.post("/tune_model")
async def tune_model_endpoint(req: TuneModelRequest):
    ml_service.tune_model(
        df_train_filename=req.df_train_filename,
        df_test_filename=req.df_test_filename,
        n_iter=req.n_iter,
        cv_folds=req.cv_folds,
        random_seed=req.random_seed
    )
    return {"msg": "Hyperparameter tuning request added to queue."}

@router.post("/lstm_train")
async def lstm_train_endpoint(req: LSTMTrainRequest):
    ml_service.lstm_train(
        train_path=req.train_path,
        lookback=req.lookback,
        horizon=req.horizon,
        hidden_size=req.hidden_size,
        num_layers=req.num_layers,
        dropout=req.dropout,
        epochs=req.epochs,
        batch_size=req.batch_size,
        lr=req.lr,
        weight_decay=req.weight_decay,
        use_subset=req.use_subset,
        subset_size=req.subset_size,
        feature_set=req.feature_set,
        custom_features=req.custom_features,
        validate_features=req.validate_features,
        generate_features=req.generate_features,
        scheduler_type=req.scheduler_type,
        use_log_prices=req.use_log_prices,
        early_stopping_patience=req.early_stopping_patience,
        load_optuna_params=req.load_optuna_params,
        load_optuna_dir=req.load_optuna_dir,
        train_all_top=req.train_all_top,
        seed=req.seed
    )
    return {"msg": "LSTM training request added to queue."}

@router.post("/lstm_test")
async def lstm_test_endpoint(req: LSTMTestRequest):
    ml_service.lstm_test(
        model_dir=req.model_dir,
        test_path=req.test_path,
        batch_size=req.batch_size
    )
    return {"msg": "LSTM testing request added to queue."}

@router.post("/lstm_prepare_data")
async def lstm_prepare_data_endpoint(req: LSTMPrepareDataRequest):
    ml_service.lstm_prepare_data(
        data_path=req.data_path,
        feature_set=req.feature_set,
        train_months=req.train_months,
        agent_months=req.agent_months,
        test_months=req.test_months,
        train_pct=req.train_pct,
        agent_pct=req.agent_pct,
        test_pct=req.test_pct,
        lookback=req.lookback,
        horizon=req.horizon,
        scaler_features=req.scaler_features,
        scaler_price=req.scaler_price,
        use_log_prices=req.use_log_prices
    )
    return {"msg": "LSTM data preparation request added to queue."}

@router.post("/lstm_tune")
async def lstm_tune_endpoint(req: LSTMTuneRequest):
    ml_service.lstm_tune(
        train_path=req.train_path,
        n_trials=req.n_trials,
        timeout=req.timeout,
        study_name=req.study_name,
        subset_size=req.subset_size,
        feature_set=req.feature_set,
        custom_features=req.custom_features,
        validate_features=req.validate_features,
        generate_features=req.generate_features,
        fixed_horizon=req.fixed_horizon
    )
    return {"msg": "LSTM hyperparameter tuning request added to queue."}