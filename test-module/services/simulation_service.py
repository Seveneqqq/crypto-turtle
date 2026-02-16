import os
import json
import gzip
import time
import sys
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from .core_logic_v2.ml_signal import ModelSignal
from .core_logic_v2.decision import DecisionEngine
from .test.rating import Rating
from .visualization_service import VisualizationService
from .core_logic_v2.structures import CoreConfig


class SimulationService:
    def __init__(self, base_dir = None):
        if base_dir is None: base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.DATA_DIR = os.path.join(base_dir, "shared", "files", "processed_simulation_data")
        self.SIM_DIR = os.path.join(base_dir, "shared", "simulation","simulations")
        os.makedirs(self.SIM_DIR, exist_ok=True)
        self.OUTPUT_DIR = os.path.join(base_dir, "shared", "simulation", "outputs")
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.rating = Rating()
        self.core_config = CoreConfig()
        self.num_ticks = 0
        self.models = {}
        self.engine = None
        self.output = {}


    def run_simulation(
        self,
        rabbit_msg_id=None,
        balance_start=1000.0,
        add_fee=True,
        sim_mode="mlrl",
        rlagent_deterministic=False,
        slippage=0,
        wallet_percentage_limit=0.9,
        max_wallet_percentage_crypto=1.0,
        stop_loss=1.0,
        threshold_buy=0.5,
        threshold_sell=0.5,
        mincost=1.0,
        tx_fraction=0.01,
        max_fraction_when_confident=0.5,
        data_files=None,
        model_paths=None,
        agent_rl=None,
        plots=True,
        logs=True,
        results=True
    ):
        try:
            self.output, self.engine, self.models, symbols = {}, None, {}, []

            if sim_mode not in ("ml","mlrl", "rl"):
                raise ValueError ("[SIMULATION ERROR] Invalid simulation mode")
            if sim_mode == "mlrl":
                raise ValueError ("[SIMULATION ERROR] mlrl is disabled for now")
            print("[Simulation] Start")
            start_time = time.time()
            data_files = data_files or {}
            model_paths = model_paths or {}

            # --- Ładowanie danych ---

            # Dodać sprawdzanie open_time
            for symbol, filename in data_files.items():
                filepath = os.path.join(self.DATA_DIR, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Brak pliku: {filepath}")
                df = pd.read_csv(filepath)
                df.dropna(inplace=True)
                len_df = len(df)
                if self.num_ticks == 0: self.num_ticks = len_df
                if len_df < self.num_ticks: self.num_ticks = len_df

                # --- Inicjalizacja modeli ---
                self.models[symbol] = ModelSignal(model_name=model_paths[symbol], data=df.copy(),base_dir=self.base_dir)
                symbols.append(symbol)

                len_pred = len(self.models[symbol].predictions)

                if len_pred < self.num_ticks: self.num_ticks = len_pred

            for symbol in symbols: self.models[symbol].process_models_data(self.num_ticks)

            self.core_config = CoreConfig(
                sim_mode=sim_mode,
                balance_start=balance_start,
                add_fee=add_fee,
                symbols=symbols,
                rlagent=sim_mode,
                rlagent_deterministic=rlagent_deterministic,
                slippage=slippage,
                wallet_percentage_limit=wallet_percentage_limit,
                max_wallet_percentage_crypto=max_wallet_percentage_crypto,
                stop_loss=stop_loss,
                threshold_buy=threshold_buy,
                threshold_sell=threshold_sell,
                mincost=mincost,
                tx_fraction=tx_fraction,
                max_fraction_when_confident=max_fraction_when_confident
            )

            # --- Inicjalizacja Portfela i Decision Engine---

            self.engine = DecisionEngine(
                agent_name=agent_rl,
                models=self.models,
                base_dir=self.base_dir,
                core_mode= "sim_ml",
                core_config= self.core_config
            )

            # --- Symulacja ---
            match sim_mode:
                case "ml": self.simulation("model")
                case "rl":
                    self.engine.core_mode = "sim_rl"
                    self.simulation("agent")
                case "mlrl":
                    self.simulation("model")
                    self.engine.core_mode = "sim_rl"
                    self.simulation("agent")

            print(f"[Simulation] Zakończono pętlę symulacyjną w {timedelta(seconds=time.time() - start_time)}")
            # --- Zapis wyników ---

            print(f"[Simulation] Zapisywanie do plików")
            self.output["files"] = {
                "models": model_paths,
                "agents": agent_rl,
                "data": data_files
            }

            self.output["config"] = self.core_config.to_dict()

            #TODO: mode simulation
            sim_mode = "agent"
            gzip_name, sim_path, gzip_path = self.build_filename()

            if logs or plots or results:
                os.makedirs(sim_path, exist_ok=True)

            with gzip.open(gzip_path, "wt", encoding="utf-8") as f:
                json.dump(self.output, f, indent=2)

            if logs: self.save_logs(sim_mode, sim_path)
            if results: self.save_results(sim_mode, sim_path)
            if plots: VisualizationService().visualize(simulation_files=gzip_name, sim_path=sim_path)

            print(f"[Simulation] Zakończono w {timedelta(seconds=time.time() - start_time)}")

            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "success",
                "message": "Successfully retrieved simulation data",
                #"model_logs_file": model_log_path,
                #"agent_logs_file": agent_log_path,
                "final_portfolio_value": self.output["agent"]["metrics"]["final_portfolio_value"],
                "sharpe_ratio": self.output["agent"]["metrics"]["sharpe_ratio"],
                "max_drawdown": self.output["agent"]["metrics"]["max_drawdown"],
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


    def simulation(self, mode):

        print("[Simulation] Liczba timestampów: ", self.num_ticks)

        for t in tqdm(range(self.num_ticks), file=sys.stdout, dynamic_ncols=True, leave=True):
            self.engine.generate_decisions(t)
            self.engine.wallet.update_history()

        crypto_ref_values = {}

        for symbol in self.core_config.symbols:
            # Mnożenie wektora przez skalar -> super szybkie w numpy
            ref_vector = (self.models[symbol].close_prices / self.models[symbol].close_prices[0]) * self.core_config.balance_start
            crypto_ref_values[symbol] = ref_vector.tolist()

        metrics = self.rating.rate(self.engine.wallet.portfolio_values_history, self.engine.wallet.transaction_count)

        serializable_logs = [order.to_dict() for order in self.engine.wallet.logs]
        self.engine.wallet.logs = serializable_logs
        self.output["hodl"] = {"crypto_ref_values": crypto_ref_values}

        self.output[mode] = {
            "timestamps": self.num_ticks,
            "metrics": metrics,
            "portfolio_values": self.engine.wallet.portfolio_values_history,
            "holdings": self.engine.wallet.portfolio_holdings_history
        }
        return

    def save_logs(self, mode, sim_path):
        log_path = os.path.join(sim_path, f"{mode}_transaction_logs.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            for line in self.engine.wallet.logs:
                f.write(json.dumps(line) + "\n")


    def save_results(self, mode, sim_path):
        results_path = os.path.join(sim_path, "sim_results.txt")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self.output[mode]["metrics"], f, indent=2)

    def build_filename(self):
        date_str = datetime.now().strftime("%d_%m")
        existing_files = [
            f for f in os.listdir(self.OUTPUT_DIR)
            if f.startswith("sim_") and f"_{date_str}_" in f and f.endswith(".gz")
        ]

        version_numbers = []
        for f in existing_files:
            try:
                base = os.path.splitext(os.path.splitext(f)[0])[0]
                v_part = base.split("_v")[-1].split("_")[0]
                version_numbers.append(int(v_part))
            except ValueError:
                continue

        next_version = max(version_numbers, default=0) + 1
        s_p = f"sim_{date_str}_v{next_version}_{self.core_config.sim_mode}"
        g_n = f"{s_p}.json.gz"
        sim_path = os.path.join(self.SIM_DIR, s_p)
        gzip_path = os.path.join(self.OUTPUT_DIR, g_n)
        return g_n, sim_path, gzip_path