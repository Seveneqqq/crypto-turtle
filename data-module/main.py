from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator
from typing import List, Union, Optional

from configuration.rabbitmq_queues import DATA_REQUEST_QUEUE
from services import rabbitmq_service
from services.realtime_data_service import RealtimeDataService
from services.data_service import DataService


app = FastAPI()
data_service = DataService()

# 🔹 inicjalizacja realtime service
realtime_data_service = RealtimeDataService(rabbitmq_service=rabbitmq_service)



# 🔹 inicjalizacja realtime service
realtime_data_service = RealtimeDataService(rabbitmq_service=rabbitmq_service)


@app.get("/")
def root():
    return {"message": "Hello from data-module"}


# =====================================================
# KONFIGURACJA DO WALIDACJI
# =====================================================

LSTM_SETS = {"set1", "set2", "set3", "set4"}
SKLEARN_SETS = {"set1", "set2"}
ALL_SETS = LSTM_SETS | SKLEARN_SETS


# =====================================================
# MODEL REQUESTU
# =====================================================

class SymbolConfig(BaseModel):
    symbol: str
    features: Union[List[str], str]  # lista feature / set / "auto"
    model: Optional[str] = None

    @model_validator(mode="after")
    def validate_logic(self):

        # ---------------- AUTO ----------------
        if self.features == "auto":
            if not self.model:
                raise ValueError("Model must be provided when features='auto'")
            return self

        # ---------------- JEŻELI NIE AUTO ----------------
        if self.model is not None:
            raise ValueError("Model can only be used when features='auto'")

        # ---------------- LISTA ----------------
        if isinstance(self.features, list):

            sets_in_list = [f for f in self.features if f in ALL_SETS]

            # tylko jeden set
            if len(sets_in_list) > 1:
                raise ValueError("Only one feature set can be provided")

            # brak mieszania set + pojedyncze feature
            if len(sets_in_list) == 1 and len(self.features) > 1:
                raise ValueError("Feature set cannot be mixed with single features")

        return self


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
            interval=request.interval
        )

        return {
            "status": "success",
            "interval": request.interval,
            "symbols": request.symbols,
            "message": "Realtime stream started"
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
                "lstm_set1": [
                    "ma_5", "ma_10", "ma_20",
                    "sma_10", "sma_30",
                    "cumvol", "volume_norm",
                    "std_20",
                ],
                "lstm_set2": [
                    "RSI_14",
                    "EMA_20",
                    "EMA_50",
                ],
                "lstm_set3": [
                    "BB_upper",
                    "BB_lower",
                    "Momentum_10",
                    "ROC_10",
                ],
                "lstm_set4": [
                    "ATR_14",
                    "VWAP",
                    "ROC_5",
                ],
            },
            "lstm_all_single_features": [
                "ma_5", "ma_10", "ma_20",
                "sma_10", "sma_30",
                "cumvol", "volume_norm",
                "std_20",
                "RSI_14",
                "EMA_20",
                "EMA_50",
                "BB_upper",
                "BB_lower",
                "Momentum_10",
                "ROC_10",
                "ATR_14",
                "VWAP",
                "ROC_5",
            ],
        },
        "sklearn": {
            "sklearn_feature_sets": {
                "sklearn_set1": [
                    "ret1",
                    "ret1_lag1",
                    "ret1_lag2",
                    "ret1_lag5",
                    "ret1_lag10",
                    "roll_mean_10",
                    "roll_mean_50",
                    "roll_std_10",
                    "roll_std_50",
                    "macd",
                    "macd_sig",
                ],
                "sklearn_set2": [
                    "ma_5",
                    "ma_10",
                    "ma_20",
                    "RSI_14",
                    "EMA_20",
                    "EMA_50",
                ],
            },
            "sklearn_all_single_features": [
                "ret1",
                "ret1_lag1",
                "ret1_lag2",
                "ret1_lag5",
                "ret1_lag10",
                "roll_mean_10",
                "roll_mean_50",
                "roll_std_10",
                "roll_std_50",
                "macd",
                "macd_sig",
                "ma_5",
                "ma_10",
                "ma_20",
                "RSI_14",
                "EMA_20",
                "EMA_50",
            ],
        },
    }


# =====================================================
# RabbitMQ Consumer
# =====================================================

rabbitmq_service.consume(
    queue=DATA_REQUEST_QUEUE,
    callback=rabbitmq_service.handle_request
)