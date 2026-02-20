import os
import json
import threading
import logging
import time
import requests
from datetime import datetime
from collections import deque
import websocket


class RealtimeDataService:
    def __init__(self, rabbitmq_service=None):
        self.rabbitmq_service = rabbitmq_service
        self.ws = None
        self.symbol = None
        self.interval = None
        self.ws_url = None
        self.is_running = False
        self.candles_buffer = deque(maxlen=100)
        self.rest_url = "https://api.binance.com/api/v3"

        self.log_dir = os.path.abspath(os.path.join("..", "realtime_data_logs"))
        os.makedirs(self.log_dir, exist_ok=True)

        main_log_path = os.path.join(
            self.log_dir,
            f"realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        buffer_log_path = os.path.join(
            self.log_dir,
            f"buffer_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        self.logger = logging.Logger(f"RealtimeLogger_{id(self)}")
        self.logger.setLevel(logging.INFO)

        main_file_handler = logging.FileHandler(main_log_path)
        main_file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )

        self.logger.addHandler(main_file_handler)
        self.logger.addHandler(console_handler)

        self.buffer_logger = logging.Logger(f"BufferLogger_{id(self)}")
        self.buffer_logger.setLevel(logging.INFO)

        buffer_file_handler = logging.FileHandler(buffer_log_path)
        buffer_file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s")
        )

        self.buffer_logger.addHandler(buffer_file_handler)

        self.logger.info("SERVICE INITIALIZED")

    def load_initial_history(self):
        url = f"{self.rest_url}/klines"
        params = {
            "symbol": self.symbol.upper(),
            "interval": self.interval,
            "limit": 100
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            self.logger.error(f"REST ERROR {response.text}")
            raise Exception(response.text)

        data = response.json()
        self.candles_buffer.clear()

        for k in data:
            candle = {
                "open_time": k[0],
                "close_time": k[6],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5])
            }
            self.candles_buffer.append(candle)

        self.logger.info(f"INITIAL BUFFER SIZE {len(self.candles_buffer)}")
        self.log_buffer_state("INITIAL LOAD")

    def on_open(self, ws):
        self.logger.info(f"WS OPEN {self.symbol} {self.interval}")

    def on_message(self, ws, message):
        data = json.loads(message)

        if data.get("e") == "kline":
            kline = data["k"]

            current_price = float(kline["c"])
            current_volume = float(kline["v"])
            is_closed = kline["x"]

            self.logger.info(
                f"LIVE UPDATE | price={current_price} | volume={current_volume} | closed={is_closed}"
            )

            if is_closed:
                new_candle = {
                    "open_time": kline["t"],
                    "close_time": kline["T"],
                    "open": float(kline["o"]),
                    "high": float(kline["h"]),
                    "low": float(kline["l"]),
                    "close": current_price,
                    "volume": current_volume
                }

                self.candles_buffer.append(new_candle)

                self.logger.info("========== CANDLE CLOSED ==========")
                for key, value in new_candle.items():
                    self.logger.info(f"{key}: {value}")
                self.logger.info(f"BUFFER SIZE: {len(self.candles_buffer)}")
                self.logger.info("===================================")

                candles_snapshot = list(self.candles_buffer)

                # ============================================================
                # 🔥 TUTAJ PODŁĄCZASZ SWÓJ SERWIS FEATURE PROCESSING
                # np:
                # processed_data = your_feature_service.process(candles_snapshot)
                #
                # NA RAZIE JEST ZAŚLEPKA:
                processed_data = candles_snapshot
                # ============================================================

                if self.rabbitmq_service:
                    payload = {
                        "event": "processed_candles",
                        "symbol": self.symbol,
                        "interval": self.interval,
                        "data": processed_data
                    }

                    self.rabbitmq_service.publish(
                        queue="decision_queue",
                        message=payload
                    )

                    self.logger.info("PUBLISHED PROCESSED DATA TO decision_queue")

                self.log_buffer_state("AFTER UPDATE")

    def log_buffer_state(self, label):
        if not self.candles_buffer:
            return

        oldest = self.candles_buffer[0]
        newest = self.candles_buffer[-1]

        oldest_dt = datetime.fromtimestamp(oldest["open_time"] / 1000)
        newest_dt = datetime.fromtimestamp(newest["open_time"] / 1000)

        self.buffer_logger.info(
            f"{label} | SIZE={len(self.candles_buffer)} | "
            f"OLDEST={oldest_dt} | NEWEST={newest_dt}"
        )

    def on_error(self, ws, error):
        self.logger.error(f"WS ERROR {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.logger.warning(f"WS CLOSED {close_status_code} {close_msg}")
        if self.is_running:
            self.reconnect()

    def connect(self):
        self.logger.info(f"WS CONNECT {self.ws_url}")

        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )

        self.ws.run_forever(ping_interval=20, ping_timeout=10)

    def start(self, symbol: str, interval: str = "1m"):
        self.symbol = symbol.lower()
        self.interval = interval
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@kline_{self.interval}"
        self.is_running = True

        self.logger.info(f"START {self.symbol} {self.interval}")

        self.load_initial_history()

        thread = threading.Thread(target=self.connect)
        thread.daemon = True
        thread.start()

    def stop(self):
        self.logger.info("STOP CALLED")
        self.is_running = False
        if self.ws:
            self.ws.close()
        self.logger.info("STREAM STOPPED")

    def reconnect(self):
        self.logger.info("RECONNECT START")
        time.sleep(5)
        self.load_initial_history()
        self.connect()