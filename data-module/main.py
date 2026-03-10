from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List, Union, Optional

from configuration.rabbitmq_queues import DATA_REQUEST_QUEUE
from services import rabbitmq_service
from services.realtime_data_service import RealtimeDataService
from services.data_service import DataService
from services.data_service_plus import PREPROCESSING_METHODS


app = FastAPI()
data_service = DataService()
realtime_data_service = RealtimeDataService(rabbitmq_service=rabbitmq_service)


@app.get("/")
def root():
    return {"message": "Hello from data-module"}


# =====================================================
# MODEL REQUESTU
# =====================================================

class SymbolConfig(BaseModel):
    symbol: str
    # wszystkie pola opcjonalne — jeśli nie podane, używany jest config z pliku
    features:       Optional[Union[List[str], str]] = None
    model:          Optional[str] = None
    preprocessing:  Optional[str] = None

    @field_validator("preprocessing")
    @classmethod
    def validate_preprocessing(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        upper = v.upper()
        if upper not in PREPROCESSING_METHODS:
            raise ValueError(
                f"Invalid preprocessing method '{v}'. "
                f"Allowed: {sorted(PREPROCESSING_METHODS)}"
            )
        return upper


class RealtimeRequest(BaseModel):
    symbols: List[SymbolConfig]
    interval: str = "1m"


# =====================================================
# ENDPOINT START STREAM
# =====================================================

@app.post("/realtime/start")
def start_realtime_stream(request: RealtimeRequest):
    try:
        realtime_data_service.start(
            symbols=request.symbols,
            interval=request.interval,
        )
        return {
            "status": "success",
            "interval": request.interval,
            "symbols": [
                {
                    "symbol":        s.symbol,
                    "features":      s.features,
                    "model":         s.model,
                    "preprocessing": s.preprocessing,
                }
                for s in request.symbols
            ],
            "message": "Realtime stream started",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# FEATURES LIST
# =====================================================

@app.get("/realtime/features-list")
def get_features_list():
    return {
        "lstm": {
            "feature_sets": {
                "lstm_set1": ["ma_5", "ma_10", "ma_20", "sma_10", "sma_30", "cumvol", "volume_norm", "std_20"],
                "lstm_set2": ["RSI_14", "EMA_20", "EMA_50"],
                "lstm_set3": ["BB_upper", "BB_lower", "Momentum_10", "ROC_10"],
                "lstm_set4": ["ATR_14", "VWAP", "ROC_5"],
            },
            "lstm_all_single_features": [
                "ma_5", "ma_10", "ma_20", "sma_10", "sma_30",
                "cumvol", "volume_norm", "std_20",
                "RSI_14", "EMA_20", "EMA_50",
                "BB_upper", "BB_lower", "Momentum_10", "ROC_10",
                "ATR_14", "VWAP", "ROC_5",
            ],
        },
        "sklearn": {
            "sklearn_feature_sets": {
                "sklearn_set1": [
                    "ret1", "ret1_lag1", "ret1_lag2", "ret1_lag5", "ret1_lag10",
                    "roll_mean_10", "roll_mean_50", "roll_std_10", "roll_std_50",
                    "macd", "macd_sig",
                ],
                "sklearn_set2": ["ma_5", "ma_10", "ma_20", "RSI_14", "EMA_20", "EMA_50"],
                "sklearn_set3": [],
                "sklearn_set4": [
                    "ret1", "ret1_lag1", "ret1_lag2", "ret1_lag5", "ret1_lag10",
                    "roll_mean_10", "roll_std_10", "roll_mean_50", "roll_std_50",
                    "macd", "macd_sig",
                    "target", "ATR_14", "VWAP", "ROC_5",
                ],
            },
            "sklearn_all_single_features": [
                "ret1", "ret1_lag1", "ret1_lag2", "ret1_lag5", "ret1_lag10",
                "roll_mean_10", "roll_mean_50", "roll_std_10", "roll_std_50",
                "macd", "macd_sig",
                "ma_5", "ma_10", "ma_20",
                "RSI_14", "EMA_20", "EMA_50",
            ],
        },
        "preprocessing_methods": sorted(PREPROCESSING_METHODS),
    }


# =====================================================
# RabbitMQ Consumer
# =====================================================

rabbitmq_service.consume(
    queue=DATA_REQUEST_QUEUE,
    callback=rabbitmq_service.handle_request,
)