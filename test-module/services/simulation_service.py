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
        self.sim_mode = "ml"
        self.balance_start = 0.0
        self.rating = Rating()
        self.rl_agents = None
        self.rlagent_deterministic = None
        self.engine = None
        self.output = {}
        self.wallet = None

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
            self.output = {}
            self.wallet = None
            if sim_mode not in ("ml","mlrl", "rl"):
                raise ValueError ("[SIMULATION ERROR] Invalid simulation mode")
            if sim_mode in ("mlrl", "rl"):
                raise ValueError ("[SIMULATION ERROR] mlrl and rl are disabled for now")
            print("[Simulation] Start")
            start_time = time.time()

            if add_fee:
                fee = 0.001
            else: fee = 0.0

            self.output["files"] = {
                "models": model_paths,
                "agents": agent_rl,
                "data": data_files
            }

            self.output["config"] = {
                "sim_mode": sim_mode,
                "balance_start": balance_start,
                "add_fee": add_fee
            }
            if sim_mode in ("mlrl", "rl"):
                rlagent=True
            else: rlagent=False
            self.sim_mode = sim_mode
            if rlagent:
                self.output["config"].update({
                    "rlagent": rlagent,
                    "rlagent_deterministic": rlagent_deterministic
                })
            else:
                self.output["config"].update({
                    "rlagent": rlagent,
                    "slippage": slippage,
                    "wallet_percentage_limit": wallet_percentage_limit,
                    "max_wallet_percentage_crypto": max_wallet_percentage_crypto,
                    "stop_loss": stop_loss,
                    "threshold_buy": threshold_buy,
                    "threshold_sell": threshold_sell,
                    "mincost": mincost,
                    "tx_fraction": tx_fraction,
                    "max_fraction_when_confident": max_fraction_when_confident
                })

            self.balance_start = balance_start
            self.rlagent_deterministic = rlagent_deterministic
            data_files = data_files or {}
            model_paths = model_paths or {}
            models, symbols  = {}, []
            # --- Ładowanie danych ---

            for symbol, filename in data_files.items():
                filepath = os.path.join(self.DATA_DIR, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Brak pliku: {filepath}")
                df = pd.read_csv(filepath)
                # --- Inicjalizacja modeli ---
                models[symbol] = ModelSignal(model_name=model_paths[symbol], data=df.copy())
                symbols.append(symbol)

            self.output["config"].update({"symbols": symbols})

            # --- Inicjalizacja portfela ---
            wallet_config = WalletConfig(
                wallet_percentage_limit=wallet_percentage_limit,
                mincost=mincost,
                slippage=slippage,
                stop_loss=stop_loss,
                fee=fee,
                max_wallet_percentage_crypto=max_wallet_percentage_crypto,
            )
            self.wallet = Wallet(balance_start, symbols, wallet_config)

            # --- Inicjalizacja DecisionEngine i RL agentów ---

            # silnik decyzyjny
            self.engine = DecisionEngine(
                agent_name=agent_rl,
                wallet=self.wallet,
                config= DecisionConfig(
                    threshold_buy=threshold_buy,
                    threshold_sell=threshold_sell,
                    tx_fraction=tx_fraction,
                    max_fraction_when_confident=max_fraction_when_confident
                ))


            # --- Symulacja ---
            match sim_mode:
                case "ml":
                    model_logs = self.simulation(symbols, models)
                case "rl":
                    agent_logs = self.simulation(symbols, models, True)
                case "mlrl":
                    model_logs = self.simulation(symbols, models)
                    wallet2 = Wallet(balance_start, symbols, wallet_config)
                    agent_logs = self.simulation(symbols, wallet2, models, False)

            # --- Zapis wyników ---
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


            if plots:
                VisualizationService().visualize(simulation_files=g_n, sim_path=sim_path)

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


    def simulation(self, symbols, models, rlagent = False):

        # --- Symulacja ---
        num_ticks = min(
            min(len(models[symbol].data) for symbol in symbols),  # długość danych
            min(len(models[symbol].predictions) for symbol in symbols)  # długość predykcji modeli
        )

        for symbol in symbols:
            df = models[symbol].data
            if len(df) > num_ticks:
                models[symbol].data = df.iloc[-num_ticks:].reset_index(drop=True)
        crypto_ref_values = {s: [] for s in symbols}
        initial_prices = {s: float(models[s].data.iloc[0]["close"]) for s in symbols}

        print("[Simulation][INFO] num_ticks", num_ticks)

        portfolio_values, timestamps, logs, holdings = [], [], [],[]

        for t in tqdm(range(num_ticks), file=sys.stdout, dynamic_ncols=True, leave=True):
                prices = {s: models[s].data.iloc[t]["close"] for s in symbols}
                prob_vec = {}
                timestamps.append(models["ETH"].data.iloc[t]["open_time"])
                for symbol in symbols:

                    prob_array = models[symbol].predictions[t]
                    prob_row = dict(zip(models[symbol].classes, prob_array))
                    prob_vec[symbol] = prob_row

                self.engine.ml_generate_decisions(prob_vec, symbols, prices, t)


            # for symbol in symbols:
            #     timestamps.append(models[symbol].data.iloc[t]["open_time"])
            #     prob_array = models[symbol].predictions[t]
            #     prob_row = dict(zip(models[symbol].classes, prob_array))

                # if rlagent:
                #     obs = build_obs(symbol, t, models[symbol].data, self.wallet, prob_array, models[symbol].classes)
                #     action, _ = self.rl_agents[symbol].predict(obs, deterministic=self.rlagent_deterministic)
                #     scaled = (action + 1.0) / 2.0  # [-1,1] -> [0,1]
                #     # decision_config = DecisionConfig(
                #     #     threshold_buy=scaled[0],
                #     #     threshold_sell=scaled[1],
                #     #     tx_fraction=scaled[2],
                #     #     max_fraction_when_confident=scaled[3],
                #     #     mincost=scaled[4] * 100
                #     # )
                #     # self.engine.config = decision_config
                #     # wallet.config.stop_loss = scaled[5]
                #     # wallet.config.max_wallet_percentage_crypto = scaled[6]
                #     self.engine.config.threshold_buy = scaled[0]
                #     self.engine.config.threshold_sell = scaled[1]
                #     self.engine.config.tx_fraction = scaled[2]

                # decision = self.engine.decide(prob_row, self.wallet, symbol, prices[symbol])
                #
                # action, qty = decision["action"], decision["qty"]

                # Wykonanie
                # if action != "hold" and qty > 0:
                #     res = self.wallet.execute_order(action, symbol, qty, prices[symbol])
                #     executed = res["executed"]
                #     action_end = res["status"] if res["status"] != "filled" else action
                # else:
                #     executed, action_end = 0.0, "hold"

                # if action_end != "hold" and executed > 0:
                #     logs.append({
                #         "symbol": symbol,
                #         "action": action_end,
                #         "timestamp": models[symbol].data.iloc[t]["open_time"],
                #         "usd_balance": f"{self.wallet.balance:.2f}",
                #         "portfel": f"{self.wallet.total_value(prices):.2f}",
                #         "reason": decision.get("reason", "")
                #     })
                #  HODL

                for symbol in symbols:
                    price_t = prices.get(symbol, None)
                    init_price = initial_prices.get(symbol, None)

                    if price_t is None or pd.isna(price_t) or init_price is None or init_price == 0:
                        crypto_ref_values[symbol].append(np.nan)
                    else:
                        hodl_value = self.balance_start * price_t / init_price
                        crypto_ref_values[symbol].append(hodl_value)

                holdings.append(self.wallet.total_holdings(prices))
                portfolio_values.append(self.wallet.total_value(prices))
        logs = self.wallet.logs
        # --- Ocena wyników ---
        metrics = self.rating.rate(portfolio_values, len(logs))


        self.output["hodl"] = {"crypto_ref_values": crypto_ref_values}

        match rlagent:
            case False:
                self.output["model"] = {
                    "timestamps": len(timestamps),
                    "metrics": metrics,
                    "portfolio_values": portfolio_values,
                    "holdings": holdings
                }
            case True:
                self.output["agent"] = {
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

        # generujemy nową nazwę
        s_p = f"sim_{date_str}_v{next_version}_{self.sim_mode}"
        g_n = f"{s_p}.json.gz"
        sim_path = os.path.join(self.SIM_DIR, s_p)
        gzip_path = os.path.join(self.OUTPUT_DIR, g_n)
        return g_n, sim_path, gzip_path



def build_obs(symbol, t, data, rl_env_wallet, prob_array, classes):
    price = float(data[symbol].iloc[t]["close"])
    prob_vec = prob_array  # numpy array
    row = data[symbol].iloc[t]
    # Mapowanie nazw klas na indeksy w tablicy
    class_idx = {c: i for i, c in enumerate(classes)}
    price_norm = price / (107312.78 + 1e-12)
    balance_norm = (float(rl_env_wallet.balance) / (1000.0 + 1e-12))
    buy_p = float(prob_vec[class_idx["buy"]])
    sell_p = float(prob_vec[class_idx["sell"]])
    hold_p = float(prob_vec[class_idx["hold"]])
    balance = float(rl_env_wallet.balance)
    ret1 = float(row.get("ret1", 0.0))
    ret1_lag1 = float(row.get("ret1_lag1", 0.0))
    roll_mean_10 = float(row.get("roll_mean_10", 0.0))
    roll_std_10 = float(row.get("roll_std_10", 0.0))
    macd = float(row.get("macd", 0.0))
    macd_sig = float(row.get("macd_sig", 0.0))
    atr = float(row.get("ATR_14", 0.0))
    vol = float(row.get("volume", 0.0))
    obs = np.array([
        buy_p, sell_p, price_norm, balance_norm,
        ret1, ret1_lag1, roll_mean_10, roll_std_10, macd, macd_sig, atr, vol
    ], dtype=np.float32)
    return obs


