import os
import json
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from services.test.wallet import Wallet, WalletConfig
from services.test.decision import DecisionEngine, DecisionConfig
from services.test.rating import Rating
from services.test.charts import Charts


class SimulationService:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(base_dir, "shared", "files", "processed_simulation_data")
        self.MODEL_DIR = os.path.join(base_dir, "shared", "models", "sklearn")
        self.SIM_DIR = os.path.join(base_dir, "shared", "simulation")
        self.AGENT_DIR = os.path.join(base_dir, "shared", "agents", "ppo")


    def run_simulation(
        self,
        rabbit_msg_id=None,
        balance_start=1000.0,
        add_fee=True,
        rlagent=True,
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
        models=None,
        agents=None,
    ):
        try:
            print("[Simulation] Start")
            start_time = time.time()

            if add_fee:
                fee = 0.001
            else: fee = 0.0

            data_files = data_files or {}
            models = models or {}
            agents = agents or {}

            # --- Ładowanie danych ---
            data = {}
            for symbol, filename in data_files.items():
                filepath = os.path.join(self.DATA_DIR, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Brak pliku: {filepath}")
                df = pd.read_csv(filepath)
                data[symbol] = df.copy()

            symbols = list(data.keys())

            # --- Inicjalizacja portfela ---
            wallet_config = WalletConfig(
                wallet_percentage_limit=wallet_percentage_limit,
                mincost=mincost,
                slippage=slippage,
                stop_loss=stop_loss,
                fee=fee,
                max_wallet_percentage_crypto=max_wallet_percentage_crypto,
            )
            wallet = Wallet(balance_start, symbols, wallet_config)
            charts = Charts()

            # --- Inicjalizacja DecisionEngine i RL agentów ---

            engines, predictions, model_outputs = {}, {}, {}

            for symbol in symbols:
                # silnik decyzyjny
                model_path = os.path.join(self.MODEL_DIR, models[symbol])
                engine = DecisionEngine(model_path=model_path, config=DecisionConfig(
                    threshold_buy=threshold_buy,
                    threshold_sell=threshold_sell,
                    tx_fraction=tx_fraction,
                    max_fraction_when_confident=max_fraction_when_confident,
                    mincost=mincost
                ))
                engines[symbol] = engine

                # predykcje
                df = data[symbol]
                features = engine.model.feature_names_in_
                x = df[features].values
                probs = engine.model.predict_proba(x)
                classes = [str(c) for c in engine.model.classes_]
                agent_classes = classes
                model_outputs[symbol] = [dict(zip(classes, row)) for row in probs]
                predictions[symbol] = probs
                agent_classes=None
                # RL agent (jeśli włączony)
                if rlagent and symbol in agents:
                    rl_agents = {}
                    agent_path = os.path.join(self.AGENT_DIR, agents[symbol])
                    if os.path.exists(agent_path):
                        rl_agents[symbol] = PPO.load(agent_path, env=None)
                    else:
                        print(f"[Simulation][WARN] Brak agenta RL dla {symbol}: {agent_path}")

            # --- Referencyjne wartości kryptowalut (HODL) i ceny początkowe ---
            model_crypto_ref_values = {s: [] for s in symbols}
            model_initial_prices = {s: float(data[s].iloc[0]["close"]) for s in symbols}
            agent_crypto_ref_values = {s: [] for s in symbols}
            agent_initial_prices = {s: float(data[s].iloc[0]["close"]) for s in symbols}

            # --- Symulacja ---
            num_ticks = min(len(df) for df in data.values())
            model_portfolio_values, model_timestamps, model_logs = [], [], []

            for t in tqdm(range(num_ticks), file=sys.stdout, dynamic_ncols=True, leave=True):
                model_prices = {s: data[s].iloc[t]["close"] for s in symbols}
                model_timestamps.append(data[symbols[0]].iloc[t]["open_time"])

                for symbol in symbols:
                    prob_array = predictions[symbol][t]
                    prob_row = dict(zip(engines[symbol].model.classes_, prob_array))

                        # klasyczny decision engine
                    decision = engines[symbol].decide(prob_row, wallet, symbol, model_prices[symbol])

                    action, qty = decision["action"], decision["qty"]

                    # Wykonanie
                    if action != "hold" and qty > 0:
                        res = wallet.execute_order(action, symbol, qty, model_prices[symbol])
                        executed = res["executed"]
                        action_end = res["status"] if res["status"] != "filled" else action
                    else:
                        executed, action_end = 0.0, "hold"

                    if action_end != "hold" and executed > 0:
                        model_logs.append({
                            "symbol": symbol,
                            "action": action_end,
                            "timestamp": data[symbol].iloc[t]["open_time"],
                            "usd_balance": f"{wallet.balance:.2f}",
                            "portfel": f"{wallet.total_value(model_prices):.2f}",
                            "reason": decision.get("reason", "")
                        })
                    # === tutaj budujemy linię HODL (wartość, gdyby cały startowy balans był zainwestowany w 'symbol') ===
                    price_t = model_prices.get(symbol, None)
                    init_price = model_initial_prices.get(symbol, None)

                    if price_t is None or pd.isna(price_t) or init_price is None or init_price == 0:
                        model_crypto_ref_values[symbol].append(np.nan)
                    else:
                        hodl_value = balance_start * price_t / init_price
                        model_crypto_ref_values[symbol].append(hodl_value)

                model_portfolio_values.append(wallet.total_value(model_prices))

            # --- Ocena wyników ---
            rating = Rating()
            model_metrics = rating.rate(model_portfolio_values,len(model_logs))
#---------------------------------------------------------------------------------------
            if rlagent:
                wallet_config = WalletConfig(
                    wallet_percentage_limit=wallet_percentage_limit,
                    mincost=mincost,
                    slippage=slippage,
                    stop_loss=stop_loss,
                    fee=fee,
                    max_wallet_percentage_crypto=max_wallet_percentage_crypto,
                )
                wallet = Wallet(balance_start, symbols, wallet_config)
                agent_portfolio_values, agent_timestamps, agent_logs = [], [], []
                for t in tqdm(range(num_ticks), file=sys.stdout, dynamic_ncols=True, leave=True):
                    agent_prices = {s: data[s].iloc[t]["close"] for s in symbols}
                    agent_timestamps.append(data[symbols[0]].iloc[t]["open_time"])

                    for symbol in symbols:
                        prob_array = predictions[symbol][t]
                        prob_row = dict(zip(engines[symbol].model.classes_, prob_array))


                        # RL agent
                        obs = build_obs(symbol, t, data, wallet, model_outputs, agent_classes)
                        action, _ = rl_agents[symbol].predict(obs, deterministic=rlagent_deterministic)
                        scaled = (action + 1.0) / 2.0  # [-1,1] -> [0,1]
                        decision_config = DecisionConfig(
                            threshold_buy=scaled[0],
                            threshold_sell=scaled[1],
                            tx_fraction=scaled[2],
                            max_fraction_when_confident=scaled[3],
                            mincost=scaled[4] * 100
                        )

                        engines[symbol].config = decision_config
                        wallet.config.stop_loss = scaled[5]
                        wallet.config.max_wallet_percentage_crypto = scaled[6]

                        decision = engines[symbol].decide(prob_row, wallet, symbol, agent_prices[symbol])


                        action, qty = decision["action"], decision["qty"]

                        # Wykonanie
                        if action != "hold" and qty > 0:
                            res = wallet.execute_order(action, symbol, qty, agent_prices[symbol])
                            executed = res["executed"]
                            action_end = res["status"] if res["status"] != "filled" else action
                        else:
                            executed, action_end = 0.0, "hold"

                        if action_end != "hold" and executed > 0:
                            agent_logs.append({
                                "symbol": symbol,
                                "action": action_end,
                                "timestamp": data[symbol].iloc[t]["open_time"],
                                "usd_balance": f"{wallet.balance:.2f}",
                                "portfel": f"{wallet.total_value(agent_prices):.2f}",
                                "reason": decision.get("reason", "")
                            })
                        # === tutaj budujemy linię HODL (wartość, gdyby cały startowy balans był zainwestowany w 'symbol') ===
                        price_t = agent_prices.get(symbol, None)
                        init_price = agent_initial_prices.get(symbol, None)

                        if price_t is None or pd.isna(price_t) or init_price is None or init_price == 0:
                            agent_crypto_ref_values[symbol].append(np.nan)
                        else:
                            hodl_value = balance_start * price_t / init_price
                            agent_crypto_ref_values[symbol].append(hodl_value)

                    agent_portfolio_values.append(wallet.total_value(agent_prices))

                # --- Ocena wyników ---

                agent_metrics = rating.rate(agent_portfolio_values, len(agent_logs))

            # --- Zapis wyników ---
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            sim_path = os.path.join(self.SIM_DIR, f"{timestamp_str}")
            os.makedirs(sim_path, exist_ok=True)

            model_log_path = os.path.join(sim_path, "model_transaction_logs.txt")
            agent_log_path = os.path.join(sim_path, "agent_transaction_logs.txt")
            model_plot_path = os.path.join(sim_path, "model_plot.png")
            agent_plot_path = os.path.join(sim_path, "agent_plot.png")
            sim_plot_path = os.path.join(sim_path, "sim_plot.png")
            results_path = os.path.join(sim_path, "sim_results.txt")

            charts.draw_chart(model_portfolio_values, model_crypto_ref_values, model_timestamps, model_metrics, symbols,
                              model_plot_path)

            if rlagent:
                charts.draw_chart(agent_portfolio_values, agent_crypto_ref_values, agent_timestamps, agent_metrics, symbols, agent_plot_path)
                charts.draw_common_chart(model_portfolio_values, agent_portfolio_values, model_timestamps, agent_timestamps, model_metrics, agent_metrics, sim_plot_path)

            with open(model_log_path, "w", encoding="utf-8") as f:
                for line in model_logs:
                    f.write(json.dumps(line) + "\n")
            if rlagent:
                print("zapisywanie do log agent")
                print(agent_logs)
                with open(agent_log_path, "w", encoding="utf-8") as f:
                    for line in agent_logs:
                        f.write(json.dumps(line) + "\n")

            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(model_metrics, f, indent=2)
                if rlagent:
                    json.dump(agent_metrics, f, indent=2)

            print(f"[Simulation] Zakończono w {timedelta(seconds=time.time() - start_time)}")

            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "success",
                "message": "Successfully retrieved simulation data",
                "model_logs_file": model_log_path,
                "agent_logs_file": agent_log_path,
                "model_plot_file": model_plot_path,
                "agent_plot_file": agent_plot_path,
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


def build_obs(symbol, t, data, rl_env_wallet, model_outputs, classes):
    price = float(data[symbol].iloc[t]["close"])
    prob_vec = model_outputs[symbol][t]

    buy_p = float(prob_vec.get("buy", 0.0))
    sell_p = float(prob_vec.get("sell", 0.0))
    hold_p = float(prob_vec.get("hold", 0.0))
    balance = float(rl_env_wallet.balance)

    obs = np.array([buy_p, sell_p, price, balance], dtype=np.float32)
    return obs
