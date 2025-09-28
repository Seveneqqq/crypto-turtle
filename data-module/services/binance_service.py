import requests
import pandas as pd
import time
import os
from datetime import datetime

class BinanceService:

    def __init__(self, api_key, api_url="https://api.binance.com/api/v3"):
        self.api_key = api_key
        self.api_url = api_url
        self.base_dir = os.getenv(
            "SHARED_DIR",
            os.path.join("shared", "files", "raw_data")
        )
        os.makedirs(self.base_dir, exist_ok=True)

    def get_klines_batch(self, symbol, interval, start_time, end_time, limit=1000):

        url = f"{self.api_url}/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }
        
        print("Params from query /klines")
        print(params)

        response = requests.get(url, params=params)
        return response.json()

    def get_historical_klines(self, rabbitmq_msg_id, symbol, interval, start_dt, end_dt, filename):

        start_ts = int(pd.to_datetime(start_dt).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_dt).timestamp() * 1000)

        all_klines = []

        while True:
            batch = self.get_klines_batch(symbol, interval, start_ts, end_ts)
            if not batch:
                break

            all_klines.extend(batch)

            last_close_time = batch[-1][6]
            start_ts = last_close_time + 1

            if start_ts >= end_ts:
                break

            time.sleep(0.2)

        columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ]
        df = pd.DataFrame(all_klines, columns=columns)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        full_path = os.path.join(self.base_dir, filename)

        df.to_csv(full_path, index=False)

        return {
            "rabbit_msg_id": rabbitmq_msg_id,
            "result": "success",
            "message": "Successfully retrieved historical data",
            "filename": filename,
            "path": full_path,
            "timestamp": datetime.now().isoformat()
        }
