import os
import json
import threading
import logging
import requests
from datetime import datetime
from collections import deque

import websocket
import pandas as pd
import numpy as np
import joblib


class RealtimeDataService:

    FEATURE_SETS = {
        "lstm_set1": [
            "ma_5", "ma_10", "ma_20",
            "sma_10", "sma_30",
            "cumvol", "volume_norm",
            "std_20",
        ],
        "lstm_set2": ["RSI_14", "EMA_20", "EMA_50"],
        "lstm_set3": ["BB_upper", "BB_lower", "Momentum_10", "ROC_10"],
        "lstm_set4": ["ATR_14", "VWAP", "ROC_5"],
        "sklearn_set1": [
            "ret1", "ret1_lag1", "ret1_lag2", "ret1_lag5", "ret1_lag10",
            "roll_mean_10", "roll_mean_50", "roll_std_10", "roll_std_50",
            "macd", "macd_sig",
        ],
        "sklearn_set2": [
            "ma_5", "ma_10", "ma_20",
            "RSI_14", "EMA_20", "EMA_50",
        ],
    }

    def __init__(self, rabbitmq_service=None):
        self.rabbitmq_service = rabbitmq_service
        self.interval = None
        self.is_running = False
        self.rest_url = "https://api.binance.com/api/v3"

        self.symbol_configs = {}
        self.models = {}
        self.buffers = {}
        self.latest_results = {}
        self.ws_by_symbol = {}
        self.ws_threads = {}

        self.publish_lock = threading.Lock()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        self.BASE_MODEL_DIR = os.path.join(project_root, "shared", "models")

        self.log_dir = os.path.abspath(os.path.join(project_root, "realtime_data_logs"))
        os.makedirs(self.log_dir, exist_ok=True)

        main_log_path = os.path.join(
            self.log_dir,
            f"realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        self.logger = logging.Logger(f"RealtimeLogger_{id(self)}")
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(main_log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info("SERVICE INITIALIZED")

    # ============================================================
    # START
    # ============================================================

    def start(self, symbols, interval="1m"):
        self.interval = interval
        self.is_running = True

        for cfg in symbols:
            symbol = str(cfg.symbol).lower()

            self.symbol_configs[symbol] = {
                "features": cfg.features,
                "model": getattr(cfg, "model", None),
            }

            self.buffers[symbol] = deque(maxlen=250)

            if cfg.features == "auto":
                model_path = self._find_model(cfg.model)
                self.models[symbol] = joblib.load(model_path)
                self.logger.info(f"[{symbol}] MODEL LOADED")

            self.load_initial_history(symbol)

            t = threading.Thread(target=self.connect, args=(symbol,), daemon=True)
            self.ws_threads[symbol] = t
            t.start()

    # ============================================================
    # FEATURE EXPAND
    # ============================================================

    def _expand_features(self, features_cfg):
        if features_cfg == "auto":
            return "auto"

        if isinstance(features_cfg, str):
            if features_cfg in self.FEATURE_SETS:
                return self.FEATURE_SETS[features_cfg]
            return [features_cfg]

        if isinstance(features_cfg, list):
            if len(features_cfg) == 1 and features_cfg[0] in self.FEATURE_SETS:
                return self.FEATURE_SETS[features_cfg[0]]
            return features_cfg

        return []

    # ============================================================
    # MODEL SEARCH
    # ============================================================

    def _find_model(self, model_name):
        for subfolder in os.listdir(self.BASE_MODEL_DIR):
            path = os.path.join(self.BASE_MODEL_DIR, subfolder, model_name)
            if os.path.exists(path):
                return path
        raise ValueError("Model not found")

    # ============================================================
    # HISTORY
    # ============================================================

    def load_initial_history(self, symbol):
        url = f"{self.rest_url}/klines"
        params = {"symbol": symbol.upper(), "interval": self.interval, "limit": 250}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()

        self.buffers[symbol].clear()

        for k in r.json():
            self.buffers[symbol].append({
                "open_time": k[0],
                "close_time": k[6],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })

        self.logger.info(f"[{symbol}] HISTORY LOADED")

    # ============================================================
    # COMPUTE FEATURES
    # ============================================================

    def _compute_features_full(self, df):
        close = df["close"]
        volume = df["volume"]

        df["ma_5"] = close.rolling(5).mean()
        df["ma_10"] = close.rolling(10).mean()
        df["ma_20"] = close.rolling(20).mean()

        df["ret1"] = np.log(close).diff()
        for k in [1, 2, 5, 10]:
            df[f"ret1_lag{k}"] = df["ret1"].shift(k)

        df["macd"] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        df["macd_sig"] = df["macd"].ewm(span=9).mean()

        df["RSI_14"] = 100 - (
            100 / (
                1 + (
                    close.diff().clip(lower=0).rolling(14).mean() /
                    (close.diff().clip(upper=0).abs().rolling(14).mean() + 1e-9)
                )
            )
        )

        df["EMA_20"] = close.ewm(span=20).mean()
        df["EMA_50"] = close.ewm(span=50).mean()

        df["cumvol"] = volume.cumsum()
        df["volume_norm"] = (volume - volume.min()) / (volume.max() - volume.min() + 1e-9)

        df.fillna(0.0, inplace=True)
        return df

    # ============================================================
    # WS
    # ============================================================

    def connect(self, symbol):
        ws_url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{self.interval}"
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=lambda ws_, msg: self.on_message(ws_, msg, symbol),
        )
        ws.run_forever()

    # ============================================================
    # MAIN MESSAGE
    # ============================================================

    def on_message(self, ws, message, symbol):
        data = json.loads(message)

        if data.get("e") != "kline":
            return

        kline = data["k"]
        if not kline["x"]:
            return

        self.buffers[symbol].append({
            "open_time": kline["t"],
            "close_time": kline["T"],
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
        })

        df = pd.DataFrame(list(self.buffers[symbol]))
        df = self._compute_features_full(df)

        cfg = self.symbol_configs[symbol]
        expanded = self._expand_features(cfg["features"])

        if expanded == "auto":
            model = self.models[symbol]
            selected_features = [f for f in model.feature_names_in_ if f != "target"]
        else:
            selected_features = expanded

        base_columns = [
            "open_time", "close_time",
            "open", "high", "low", "close", "volume"
        ]

        final_columns = base_columns + selected_features
        final_columns = [c for c in final_columns if c in df.columns]

        df_last = df.tail(100)[final_columns].copy()

        self.latest_results[symbol] = {
            "features_used": selected_features,
            "realtime_data": df_last.to_dict(orient="records"),
        }

        payload = {
            "result": [
                {"symbol": s, "data": self.latest_results[s]}
                for s in sorted(self.latest_results.keys())
            ]
        }

        with self.publish_lock:
            if self.rabbitmq_service:
                self.rabbitmq_service.publish(
                    queue="decision_queue",
                    message=payload
                )

            self.logger.info(
                f"[{symbol}] PAYLOAD SENT | candles=100 | features={selected_features}"
            )
