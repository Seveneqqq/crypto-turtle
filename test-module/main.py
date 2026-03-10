from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from configuration.rabbitmq_queues import TEST_REQUEST_QUEUE
from services import rabbitmq_service

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from test-module"}

# -------------------------
# ENDPOINTY BINANCE TESTNET
# -------------------------

from services.core_logic.binance_testnet_service import BinanceTestnetService

# Po realtime_data_service dodaj:
binance_testnet_service = BinanceTestnetService()

# Modele requestów (po RealtimeRequest)
class BuyRequest(BaseModel):
    symbol: str = "BTCUSDT"
    qty: float = 0.001

class SellRequest(BaseModel):
    symbol: str = "BTCUSDT"
    qty: float = 0.001
@app.post("/testnet/buy")
def buy_crypto(request: BuyRequest):
    try:
        result = binance_testnet_service.buy(symbol=request.symbol, qty=request.qty)
        if result.get("result") == "success":
            return {"status": "success", **result}
        return {"status": "failed", "message": result.get("message", "Unknown error")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/testnet/sell")
def sell_crypto(request: SellRequest):
    try:
        result = binance_testnet_service.sell(symbol=request.symbol, qty=request.qty)
        if result.get("result") == "success":
            return {"status": "success", **result}
        return {"status": "failed", "message": result.get("message", "Unknown error")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/testnet/portfolio")
def get_portfolio():
    try:
        result = binance_testnet_service.get_portfolio()
        if result.get("result") == "success":
            return {"status": "success", **result}
        return {"status": "failed", "message": result.get("message", "Unknown error")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

rabbitmq_service.consume(queue=TEST_REQUEST_QUEUE, callback = rabbitmq_service.handle_request)
rabbitmq_service.consume(queue="decision_queue", callback = rabbitmq_service.handle_realtime_request)

