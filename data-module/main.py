from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator
from typing import List, Union, Optional

from configuration.rabbitmq_queues import DATA_REQUEST_QUEUE
from services import rabbitmq_service
from services.realtime_data_service import RealtimeDataService
from services.data_service import DataService
from services.operation_history_service import OperationHistoryService
from services.operation_file_logger_service import OperationFileLoggerService

# NOWE IMPORTY
from services.optuna_dashboard_service import OptunaDashboardService
from services.optuna_results_loader_service import OptunaResultsLoaderService


app = FastAPI()

# ============================================================
# CORS
# ============================================================

origins = [
    "http://localhost:5173",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# STATIC FILES (REPORTS)
# ============================================================

app.mount(
    "/shared/files",
    StaticFiles(directory="shared/files"),
    name="shared-files"
)

data_service = DataService()
operation_history_service = OperationHistoryService()
operation_file_logger_service = OperationFileLoggerService()

realtime_data_service = RealtimeDataService(rabbitmq_service=rabbitmq_service)

# NOWA INICJALIZACJA
optuna_dashboard_service = OptunaDashboardService()
optuna_results_loader_service = OptunaResultsLoaderService()


@app.get("/")
def root():
    return {"message": "Hello from data-module"}


LSTM_SETS = {"lstm_set1", "lstm_set2", "lstm_set3", "lstm_set4"}
SKLEARN_SETS = {"sklearn_set1", "sklearn_set2"}
ALL_SETS = LSTM_SETS | SKLEARN_SETS


@app.get("/history/operations")
def get_operations(
        service_name: str | None = None,
        operation_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0
):

    rows = operation_history_service.list_operations(
        service_name=service_name,
        operation_name=operation_name,
        status=status,
        limit=limit,
        offset=offset
    )

    total = operation_history_service.count_operations(
        service_name=service_name,
        operation_name=operation_name,
        status=status
    )

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": rows
    }


class SymbolConfig(BaseModel):
    symbol: str
    features: Union[List[str], str]
    model: Optional[str] = None

    @model_validator(mode="after")
    def validate_logic(self):
        if self.features == "auto":
            if not self.model:
                raise ValueError("Model must be provided when features='auto'")
            return self

        if self.model is not None:
            raise ValueError("Model can only be used when features='auto'")

        if isinstance(self.features, list):
            sets_in_list = [f for f in self.features if f in ALL_SETS]

            if len(sets_in_list) > 1:
                raise ValueError("Only one feature set can be provided")

            if len(sets_in_list) == 1 and len(self.features) > 1:
                raise ValueError("Feature set cannot be mixed with single features")

        return self


class RealtimeRequest(BaseModel):
    symbols: List[SymbolConfig]
    interval: str = "1m"


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


# ============================================================
# NOWE ENDPOINTY ANALITYCZNE OPTUNA
# ============================================================

@app.post("/analytics/optuna/dashboard")
def generate_optuna_dashboard():
    """
    Generuje wykresy i raport z eksperymentów Optuna
    """
    try:
        result = optuna_dashboard_service.generate_dashboard()

        return {
            "status": "success",
            "message": "Optuna dashboard generated",
            "data": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/optuna/data")
def get_optuna_data():
    try:

        df = optuna_results_loader_service.load_all_results()

        df = df.replace({float("nan"): None})

        return {
            "rows": df.to_dict(orient="records"),
            "total": len(df)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/optuna/reports")
def list_optuna_reports():
    """
    Lista wygenerowanych raportów
    """
    import os

    reports_dir = "shared/files/optuna_reports"

    if not os.path.exists(reports_dir):
        return {"reports": []}

    return {
        "reports": os.listdir(reports_dir)
    }


# ============================================================
# RABBITMQ CONSUMER
# ============================================================

rabbitmq_service.consume(
    queue=DATA_REQUEST_QUEUE,
    callback=rabbitmq_service.handle_request,
)