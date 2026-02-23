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


        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = os.path.join(self.base_dir,"shared", "realtime_simulation")
        os.makedirs(self.data_dir, exist_ok=True)
        self.data_file_path = os.path.join(self.data_dir, f"realtime_data_{current_time}.csv")
        self.orders_file_path = os.path.join(self.data_dir, f"realtime_orders_{current_time}.csv")

        self.models = {}
        self.model_paths = {}
        self.symbols = []
        self.core_config = None
        self.engine = None
    def live_decision(
        self,
        rabbit_msg_id=None,
        result=None,
    ):
        try:
            x = result["candles"][-1]
            symbol = result["symbol"]

            x["open_time"] = pd.to_datetime(x["open_time"], unit="ms")

            if self.models == {} or symbol not in self.symbols:
                if symbol == "btcusdt":
                    self.models[symbol] = ModelSignal(model_name="btc_model_c_v1_1210_11.pkl", data=[],base_dir=self.base_dir, realtime=True)
                if symbol == "ethusdt":
                    self.models[symbol] = ModelSignal(model_name="eth_model_c_v1_1210_11.pkl", data=[],base_dir=self.base_dir, realtime=True)
                self.symbols.append(symbol)


            self.models[symbol].realtime_sklearn(x)

            # Silnik decyzyjny i generowanie decyzji
            if self.core_config is None:
                self.core_config = CoreConfig(
                    sim_mode="ml",
                    balance_start=1000.0,
                    add_fee=True,
                    symbols=self.symbols,
                    rlagent_deterministic=True,
                    slippage=0,
                    wallet_percentage_limit=0.8530,
                    max_wallet_percentage_crypto=0.8278,
                    stop_loss=1.0,
                    threshold_buy=0.3936,
                    threshold_sell=0.4978,
                    mincost=8.0598,
                    tx_fraction=0.0220,
                    max_fraction_when_confident=0.5098,
                    experiment_version="v0"
                )

            # --- Inicjalizacja Portfela i Decision Engine---
            if self.engine is None:
                self.engine = DecisionEngine(
                    agent_name=None,
                    models=self.models,
                    base_dir=self.base_dir,
                    core_mode="sim_ml",
                    core_config=self.core_config
                )

            self.engine.generate_decisions(t=-1)
            order = self.engine.wallet.update_history(logs=True)
            df_to_save = pd.DataFrame([x])
            order_to_save = pd.DataFrame([order])


            file_exists = os.path.isfile(self.data_file_path)
            file_exists2 = os.path.isfile(self.data_file_path)

            df_to_save.to_csv(self.data_file_path, mode='a', index=False, header=not file_exists)
            order_to_save.to_csv(self.orders_file_path, mode='a', index=False, header=not file_exists2)

            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "success",
                "message": "Successfully retrieved simulation data",
                "portfolio_values": "portfolio_values",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            import traceback
            print("[Simulation][ERROR]", e)
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

