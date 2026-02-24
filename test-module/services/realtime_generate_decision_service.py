import json
import os
import pandas as pd
from datetime import datetime, timedelta
from .core_logic.ml_signal import ModelSignal
from .core_logic.decision import DecisionEngine
from .core_logic.structures import CoreConfig, Order

class RealtimeGenerateDecisionService:
    def __init__(self, base_dir = None):
        if base_dir is None: base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.symbols = []
        self.model_paths = {}
        self.core_config = CoreConfig()

        self.config_path = "/app/configuration/realtime_simulation_config.json"
        self._last_config_mtime = 0
        self._check_and_load_config()


        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = os.path.join(self.base_dir,"shared", "realtime_simulation")
        os.makedirs(self.data_dir, exist_ok=True)
        self.data_file_path = os.path.join(self.data_dir, f"realtime_data_{current_time}.csv")
        self.orders_file_path = os.path.join(self.data_dir, f"realtime_orders_{current_time}.csv")

        self.models = {}

        self.engine = DecisionEngine(
            agent_name=None,
            models=self.models,
            base_dir=self.base_dir,
            core_mode="sim_ml",
            core_config=self.core_config
        )

    def live_decision(
        self,
        rabbit_msg_id=None,
        results=None,
        result=None,
    ):
        try:
            # for result in results:
            #     data = result["candles"][-1]
            #     symbol = result["symbol"]
            #     data["open_time"] = pd.to_datetime(data["open_time"], unit="ms")
            #     print(symbol)
            #
            #     if symbol not in self.symbols:
            #         self.models[symbol] = ModelSignal(model_name="eth_model_c_v1_1210_11.pkl", data=[],
            #                                           base_dir=self.base_dir, realtime=True)
            #         self.symbols.append(symbol)
            #
            #     self.models[symbol].generate_prediction(data)

            self._check_and_load_config()

            data = result["candles"][-1]
            symbol = result["symbol"]
            data["open_time"] = pd.to_datetime(data["open_time"], unit="ms")

            if symbol not in self.symbols:
                self.models[symbol] = ModelSignal(model_name=self.model_paths[symbol], data=[],
                                                  base_dir=self.base_dir, realtime=True)
                self.symbols.append(symbol)
                self.engine.wallet.initialize_symbol_and_holdings(symbol)

            self.models[symbol].generate_prediction(data)

            self.engine.generate_decisions(t=-1)
            order = self.engine.wallet.update_history(logs=True)

            df_to_save = pd.DataFrame([data])
            order_to_save = pd.DataFrame([order])


            file_exists = os.path.isfile(self.data_file_path)
            file_exists2 = os.path.isfile(self.data_file_path)

            df_to_save.to_csv(self.data_file_path, mode='a', index=False, header=not file_exists)
            order_to_save.to_csv(self.orders_file_path, mode='a', index=False, header=not file_exists2)

            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "success",
                "message": f"Successfully generate decision for timestamp: {datetime.now().isoformat()}",
                "portfolio_values": "portfolio_values",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            import traceback
            print("[Realtime Generate Decision][ERROR]", e)
            traceback.print_exc()
            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "failed",
                "message": str(e),
                "logs_file": None,
                "plot_file": None,
                "portfolio_values": [],
                "timestamp": datetime.now().isoformat()
            }

    def _check_and_load_config(self):
        """Sprawdza metadane pliku i przeładowuje konfigurację gdy nastąpiła zmiana."""
        try:
            if not os.path.exists(self.config_path):
                return

            current_mtime = os.path.getmtime(self.config_path)

            if current_mtime <= self._last_config_mtime:
                return

            print(f"[RealtimeGenerateDecision] Wykryto zmianę w konfiguracji. Przeładowuję...")
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
                self.model_paths = config_data.get("models", {})
                self.load_models()

                raw_core = config_data.get("core_config", {})
                if raw_core:
                    self.load_core_config(raw_core)

            self._last_config_mtime = current_mtime

        except Exception as e:
            print(f"[RealtimeGenerateDecision] Błąd sprawdzania pliku: {e}")

    def load_models(self):
        for symbol in self.symbols:
            self.models[symbol].model_name = self.model_paths[symbol]
            self.models[symbol].check_model_type()

    def load_core_config(self, raw_core):

        new_core = CoreConfig(**raw_core)
        new_core.symbols = self.symbols
        self.core_config = new_core
        if self.engine:
            self.engine.config.set(self.core_config)

            self.engine.experiment_version = self.core_config.experiment_version
            self.engine.apply_experiment_version()

            if hasattr(self.engine, 'wallet'):
                self.engine.wallet.config.set(self.core_config)
        else: pass