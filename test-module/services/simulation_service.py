import os
import json
import gzip
import time
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from .test.wallet import Wallet, WalletConfig
from .test.ml_signal import ModelSignal
from .test.decision import DecisionEngine, DecisionConfig
from .test.rating import Rating
from .visualization_service import VisualizationService


class SimulationService:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(base_dir, "shared", "files", "processed_simulation_data")
        self.SIM_DIR = os.path.join(base_dir, "shared", "simulation","simulations")
        os.makedirs(self.SIM_DIR, exist_ok=True)
        self.AGENT_DIR = os.path.join(base_dir, "shared", "agents", "ppo")
        self.OUTPUT_DIR = os.path.join(base_dir, "shared", "simulation", "outputs")
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.rating = Rating()

        self.sim_mode = "ml"
        self.balance_start = 0.0
        self.models = {}
        self.engine = None
        self.wallet = None
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
            self.output, self.engine, self.wallet, self.models, symbols = {}, None, None, {}, []

            if sim_mode not in ("ml","mlrl", "rl"):
                raise ValueError ("[SIMULATION ERROR] Invalid simulation mode")
            if sim_mode in ("mlrl", "rl"):
                raise ValueError ("[SIMULATION ERROR] mlrl and rl are disabled for now")
            print("[Simulation] Start")
            start_time = time.time()

            data_files = data_files or {}
            model_paths = model_paths or {}

            if add_fee: fee = 0.001
            else: fee = 0.0

            if sim_mode in ("mlrl", "rl"):
                rlagent=True
            else: rlagent=False

            # --- Ładowanie danych ---

            for symbol, filename in data_files.items():
                filepath = os.path.join(self.DATA_DIR, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Brak pliku: {filepath}")
                df = pd.read_csv(filepath)
                # --- Inicjalizacja modeli ---
                self.models[symbol] = ModelSignal(model_name=model_paths[symbol], data=df.copy())
                symbols.append(symbol)

            self.sim_mode = sim_mode
            self.balance_start = balance_start

            # --- Inicjalizacja Portfela i Decision Engine---

            self.wallet = Wallet(
                balance = balance_start,
                symbols = symbols,
                config  = WalletConfig(
                    wallet_percentage_limit=wallet_percentage_limit,
                    mincost=mincost,
                    slippage=slippage,
                    stop_loss=stop_loss,
                    fee=fee,
                    max_wallet_percentage_crypto=max_wallet_percentage_crypto
                )
            )

            self.engine = DecisionEngine(
                agent_name=agent_rl,
                rlagent_deterministic=rlagent_deterministic,
                wallet=self.wallet,
                config= DecisionConfig(
                    threshold_buy=threshold_buy,
                    threshold_sell=threshold_sell,
                    tx_fraction=tx_fraction,
                    max_fraction_when_confident=max_fraction_when_confident
                )
            )

            # --- Symulacja ---

            match sim_mode:
                case "ml":
                    model_logs = self.simulation(symbols)
                case "rl":
                    agent_logs = self.simulation(symbols, True)
                case "mlrl":
                    model_logs = self.simulation(symbols)
                    wallet2 = Wallet(balance_start, symbols, wallet_config)
                    agent_logs = self.simulation(symbols, wallet2, False)

            # --- Zapis wyników ---

            self.output["files"] = {
                "models": model_paths,
                "agents": agent_rl,
                "data": data_files
            }

            self.output["config"] = {
                "sim_mode": sim_mode,
                "balance_start": balance_start,
                "add_fee": add_fee,
                "symbols": symbols,
                "rlagent": rlagent,
                "rlagent_deterministic": rlagent_deterministic,
                "slippage": slippage,
                "wallet_percentage_limit": wallet_percentage_limit,
                "max_wallet_percentage_crypto": max_wallet_percentage_crypto,
                "stop_loss": stop_loss,
                "threshold_buy": threshold_buy,
                "threshold_sell": threshold_sell,
                "mincost": mincost,
                "tx_fraction": tx_fraction,
                "max_fraction_when_confident": max_fraction_when_confident,
            }

            g_n, sim_path, gzip_path = self.build_filename()

            if logs or plots or results:
                os.makedirs(sim_path, exist_ok=True)

            model_log_path = os.path.join(sim_path, "model_transaction_logs.txt")
            agent_log_path = os.path.join(sim_path, "agent_transaction_logs.txt")

            results_path = os.path.join(sim_path, "sim_results.txt")

            if sim_mode in ("ml", "mlrl"):
                if logs:
                    with open(model_log_path, "w", encoding="utf-8") as f:
                        for line in model_logs:
                            f.write(json.dumps(line) + "\n")
                if results:
                    with open(results_path, "w", encoding="utf-8") as f:
                        json.dump(self.output["model"]["metrics"], f, indent=2)

            if sim_mode in ("rl", "mlrl"):
                if logs:
                    with open(agent_log_path, "w", encoding="utf-8") as f:
                        for line in agent_logs:
                            f.write(json.dumps(line) + "\n")
                if results:
                    with open(results_path, "w", encoding="utf-8") as f:
                        json.dump(self.output["agent"]["metrics"], f, indent=2)

            with gzip.open(gzip_path, "wt", encoding="utf-8") as f:
                json.dump(self.output, f, indent=2)


            if plots: VisualizationService().visualize(simulation_files=g_n, sim_path=sim_path)

            print(f"[Simulation] Zakończono w {timedelta(seconds=time.time() - start_time)}")

            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "success",
                "message": "Successfully retrieved simulation data",
                "model_logs_file": model_log_path,
                "agent_logs_file": agent_log_path,
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


    def simulation(self, symbols, rlagent = False):

        crypto_ref_values, initial_prices, portfolio_values, timestamps, logs, holdings = {}, [], [], [], [], []
        num_ticks = min(
            min(len(self.models[symbol].data) for symbol in symbols),  # długość danych
            min(len(self.models[symbol].predictions) for symbol in symbols)  # długość predykcji modeli
        )

        for symbol in symbols:
            df = self.models[symbol].data
            if len(df) > num_ticks:
                self.models[symbol].data = df.iloc[-num_ticks:].reset_index(drop=True)
            initial_prices[symbol] = self.models[symbol].data.iloc[0]["close"]
            crypto_ref_values[symbol] = []

        print("[Simulation][INFO] num_ticks", num_ticks)

        for t in tqdm(range(num_ticks), file=sys.stdout, dynamic_ncols=True, leave=True):
            prob_vec, prices = {}, {}
            timestamps.append(self.models[symbols[0]].data.iloc[t]["open_time"])

            for symbol in symbols:
                prices[symbol] = self.models[symbol].data.iloc[t]["close"]
                prob_array = self.models[symbol].predictions[t]
                prob_row = dict(zip(self.models[symbol].classes, prob_array))
                prob_vec[symbol] = prob_row
                crypto_ref_values[symbol].append(self.balance_start * prices[symbol] / initial_prices[symbol])

            self.engine.ml_generate_decisions(prob_vec, symbols, prices, t)
            holdings.append(self.wallet.total_holdings(prices))
            portfolio_values.append(self.wallet.total_value(prices))

        logs = self.wallet.logs
        metrics = self.rating.rate(portfolio_values, len(logs))

        self.output["hodl"] = {"crypto_ref_values": crypto_ref_values}

        if rlagent: a="agent"
        else:a="model"

        self.output[a] = {
            "timestamps": len(timestamps),
            "metrics": metrics,
            "portfolio_values": portfolio_values,
            "holdings": holdings
        }

        return logs

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
        s_p = f"sim_{date_str}_v{next_version}_{self.sim_mode}"
        g_n = f"{s_p}.json.gz"
        sim_path = os.path.join(self.SIM_DIR, s_p)
        gzip_path = os.path.join(self.OUTPUT_DIR, g_n)
        return g_n, sim_path, gzip_path