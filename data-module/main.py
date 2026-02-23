from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from configuration.rabbitmq_queues import DATA_REQUEST_QUEUE
from services import rabbitmq_service
from services.realtime_data_service import RealtimeDataService

#from services.data_service import DataService

from shared.plgrid.data_module.data_service import DataService

app = FastAPI()
data_service = DataService()

# 🔹 inicjalizacja realtime service
realtime_data_service = RealtimeDataService(rabbitmq_service=rabbitmq_service, data_service=data_service)


@app.get("/")
def root():
    return {"message": "Hello from data-module"}


# -------------------------
# MODEL REQUESTU
# -------------------------
class RealtimeRequest(BaseModel):
    symbol: str
    interval: str = "1m"


# -------------------------
# ENDPOINT START STREAM
# -------------------------
@app.post("/realtime/start")
def start_realtime_stream(request: RealtimeRequest):
    try:
        realtime_data_service.start(
            symbol=request.symbol,
            interval=request.interval
        )

        return {
            "status": "success",
            "symbol": request.symbol,
            "interval": request.interval,
            "message": "Realtime stream started"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# RabbitMQ Consumer
# -------------------------
rabbitmq_service.consume(
    queue=DATA_REQUEST_QUEUE,
    callback=rabbitmq_service.handle_request
)