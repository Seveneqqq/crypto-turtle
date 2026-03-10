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

from .data_service_plus import DataServicePlus


class RealtimeDataService:

    FEATURE_SETS = {
        "lstm_set1": ["ma_5", "ma_10", "ma_20", "sma_10", "sma_30", "cumvol", "volume_norm", "std_20"],
        "lstm_set2": ["RSI_14", "EMA_20", "EMA_50"],
        "lstm_set3": ["BB_upper", "BB_lower", "Momentum_10", "ROC_10"],
        "lstm_set4": ["ATR_14", "VWAP", "ROC_5"],
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
    }

    def __init__(self, rabbitmq_service=None):
        self.rabbitmq_service = rabbitmq_service
        self.interval = None
        self.is_running = False
        self.rest_url = "https://api.binance.com/api/v3"

        self.symbol_configs = {}
        self.models = {}         # symbol -> loaded model object
        self.scalers = {}        # symbol -> (fitted_scaler | None, cols_to_scale)
        self.buffers = {}
        self.latest_results = {}
        self.ws_by_symbol = {}
        self.ws_threads = {}

        self._data_service = DataServicePlus()
        self.publish_lock = threading.Lock()

        current_dir  = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        project_root2 = os.path.abspath(os.path.join(current_dir,  ".."))
        self.BASE_MODEL_DIR = os.path.join(project_root2, "shared", "models")
        self.config_path    = os.path.join(project_root2, "configuration", "realtime_simulation_config.json")
        self._last_config_mtime = 0.0

        self.log_dir = os.path.abspath(os.path.join(project_root, "realtime_data_logs"))
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = logging.Logger(f"RealtimeLogger_{id(self)}")
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.info("SERVICE INITIALIZED")

    # ============================================================
    # START
    # ============================================================

    def start(self, symbols: list, interval: str = "1m"):
        self.interval = interval
        self.is_running = True

        # wczytaj config z pliku
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                config_data = json.load(f)
            self._last_config_mtime = os.path.getmtime(self.config_path)
        else:
            config_data = {}
            self.logger.warning(f"Config file not found: {self.config_path}")

        for cfg in symbols:
            symbol     = str(cfg.symbol).lower()
            file_cfg   = config_data.get("symbols", {}).get(symbol.upper(), {})

            # request nadpisuje config — None oznacza "weź z pliku"
            features      = cfg.features      if cfg.features      is not None else file_cfg.get("features", [])
            model         = cfg.model         if cfg.model         is not None else file_cfg.get("model", None)
            preprocessing = cfg.preprocessing if cfg.preprocessing is not None else file_cfg.get("preprocessing", "RAW")

            self.symbol_configs[symbol] = {
                "features":      features,
                "model":         model,
                "preprocessing": preprocessing.upper(),
                "model_type":    None,
            }

            self.buffers[symbol] = deque(maxlen=250)

            if model:
                self._load_model_for_symbol(symbol, model)

            self.load_initial_history(symbol)
            self._fit_scaler(symbol, self.symbol_configs[symbol]["preprocessing"])

            t = threading.Thread(target=self.connect, args=(symbol,), daemon=True)
            self.ws_threads[symbol] = t
            t.start()

    # ============================================================
    # CONFIG HOT-RELOAD
    # ============================================================

    def _check_and_load_config(self):
        """Sprawdza czy config się zmienił i przeładowuje model/features/preprocessing per symbol."""
        try:
            if not os.path.exists(self.config_path):
                return
            current_mtime = os.path.getmtime(self.config_path)
            if current_mtime <= self._last_config_mtime:
                return

            self.logger.info("Config changed — reloading...")
            with open(self.config_path, "r") as f:
                config_data = json.load(f)

            for symbol, cfg in config_data.get("symbols", {}).items():
                symbol = symbol.lower()
                if symbol not in self.symbol_configs:
                    continue

                new_features      = cfg.get("features", self.symbol_configs[symbol]["features"])
                new_preprocessing = cfg.get("preprocessing", self.symbol_configs[symbol]["preprocessing"]).upper()
                new_model         = cfg.get("model", self.symbol_configs[symbol]["model"])

                self.symbol_configs[symbol]["features"]      = new_features
                self.symbol_configs[symbol]["preprocessing"] = new_preprocessing
                self.symbol_configs[symbol]["model"]         = new_model

                if new_features == "auto" and new_model:
                    self._load_model_for_symbol(symbol, new_model)

                self._fit_scaler(symbol, new_preprocessing)
                self.logger.info(
                    f"[{symbol}] Config reloaded | model={new_model} | "
                    f"features={new_features} | preprocessing={new_preprocessing}"
                )

            self._last_config_mtime = current_mtime

        except Exception as e:
            self.logger.error(f"Config reload error: {e}")

    # ============================================================
    # LOAD MODEL
    # ============================================================

    def _load_model_for_symbol(self, symbol: str, model_name: str):
        """
        Szuka modelu po nazwie w podfolderach BASE_MODEL_DIR.
        Subfolder decyduje o typie: sklearn -> joblib.load, lstm -> (TODO).
        """
        for subfolder in os.listdir(self.BASE_MODEL_DIR):
            subfolder_path = os.path.join(self.BASE_MODEL_DIR, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            candidate = os.path.join(subfolder_path, model_name)
            if not os.path.exists(candidate):
                continue

            model_type = subfolder.lower()
            if model_type == "sklearn":
                self.models[symbol] = joblib.load(candidate)
                self.logger.info(f"[{symbol}] sklearn model loaded: {model_name}")
            elif model_type == "lstm":
                # TODO: implementacja ładowania LSTM
                pass

            self.symbol_configs[symbol]["model_type"] = model_type
            return

        raise ValueError(
            f"Model '{model_name}' not found in any subfolder of {self.BASE_MODEL_DIR}. "
            f"Available: {os.listdir(self.BASE_MODEL_DIR)}"
        )

    # ============================================================
    # FEATURE EXPAND
    # ============================================================

    def _expand_features(self, symbol: str) -> list[str] | str:
        """
        Zwraca listę feature names dla symbolu.
        Dla 'auto' + sklearn: bierze z model.feature_names_in_.
        Dla 'auto' + lstm: (TODO).
        Dla setu/listy: rozwijanie przez FEATURE_SETS.
        """
        cfg           = self.symbol_configs[symbol]
        features_cfg  = cfg["features"]
        model_type    = cfg.get("model_type")

        if features_cfg == "auto":
            if model_type == "sklearn" and symbol in self.models:
                return list(self.models[symbol].feature_names_in_)
            elif model_type == "lstm":
                # TODO: lstm features
                return []
            return []

        if isinstance(features_cfg, str):
            return self.FEATURE_SETS.get(features_cfg, [features_cfg])

        if isinstance(features_cfg, list):
            if len(features_cfg) == 1 and features_cfg[0] in self.FEATURE_SETS:
                return self.FEATURE_SETS[features_cfg[0]]
            return features_cfg

        return []

    # ============================================================
    # FIT SCALER
    # ============================================================

    def _fit_scaler(self, symbol: str, method: str):
        if method in ("RAW", "ROLLING_Z"):
            self.scalers[symbol] = (None, [])
            self.logger.info(f"[{symbol}] SCALER: {method} — no fit needed")
            return

        feature_cols = self._expand_features(symbol)
        if not feature_cols:
            self.scalers[symbol] = (None, [])
            return

        result = self._data_service.fit_scaler_from_file(
            symbol=symbol,
            method=method,
            feature_cols=feature_cols,
        )

        if result is None:
            self.scalers[symbol] = (None, [])
        else:
            scaler, cols_fitted = result
            self.scalers[symbol] = (scaler, cols_fitted)
            self.logger.info(
                f"[{symbol}] SCALER FITTED | method={method} | n_features={len(cols_fitted)}"
            )

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
                "open_time":  k[0],
                "close_time": k[6],
                "open":       float(k[1]),
                "high":       float(k[2]),
                "low":        float(k[3]),
                "close":      float(k[4]),
                "volume":     float(k[5]),
            })

        self.logger.info(f"[{symbol}] HISTORY LOADED ({len(self.buffers[symbol])} candles)")

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

        # --- hot-reload config ---
        self._check_and_load_config()

        self.buffers[symbol].append({
            "open_time":  kline["t"],
            "close_time": kline["T"],
            "open":       float(kline["o"]),
            "high":       float(kline["h"]),
            "low":        float(kline["l"]),
            "close":      float(kline["c"]),
            "volume":     float(kline["v"]),
        })

        selected_features = self._expand_features(symbol)
        method            = self.symbol_configs[symbol]["preprocessing"]

        # --- oblicz tylko potrzebne cechy ---
        df = pd.DataFrame(list(self.buffers[symbol]))
        df = DataServicePlus.compute_features_for_realtime(df, required_features=selected_features)

        base_columns  = ["open_time", "close_time", "open", "high", "low", "close", "volume"]
        final_columns = base_columns + [f for f in selected_features if f not in base_columns]
        final_columns = [c for c in final_columns if c in df.columns]

        df_last = df.tail(100)[final_columns].copy()

        # --- preprocessing ---
        scaler, cols_to_scale = self.scalers.get(symbol, (None, []))
        df_last = DataServicePlus.apply_preprocessing_realtime(
            df=df_last,
            method=method,
            scaler=scaler,
            cols_to_scale=cols_to_scale,
        )

        # --- akumuluj i publikuj gdy wszystkie symbole gotowe ---
        self.latest_results[symbol] = {
            "close_time": df_last.iloc[-1]["close_time"],
            "data": {
                "features_used": selected_features,
                "preprocessing": method,
                "realtime_data": df_last.to_dict(orient="records"),
            }
        }

        all_timestamps = [res["close_time"] for res in self.latest_results.values()]
        if len(self.latest_results) == len(self.symbol_configs) and len(set(all_timestamps)) == 1:
            payload = {
                "result": [
                    {"symbol": s, "data": self.latest_results[s]["data"]}
                    for s in sorted(self.latest_results.keys())
                ]
            }
            with self.publish_lock:
                if self.rabbitmq_service:
                    self.rabbitmq_service.publish(queue="decision_queue", message=payload)
                self.logger.info(
                    f"[{symbol}] PAYLOAD SENT | candles=100 | "
                    f"features={selected_features} | preprocessing={method}"
                )