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


class SimulationService:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(base_dir, "shared", "files", "processed_simulation_data")
        self.MODEL_DIR = os.path.join(base_dir, "shared", "models", "sklearn")
        self.SIM_DIR = os.path.join(base_dir, "shared", "simulation")
        self.AGENT_DIR = os.path.join(base_dir, "shared", "agents", "ppo")


    def run_simulation(
        self,
        rabbitmq_msg_id=None,
        balance_start=1000.0,
        add_fee=True,
        rlagent=True,
        rlagent_deterministic=False,
        slippage=0,
        wallet_percentage_limit=0.9,
        max_wallet_percentage_crypto=1.0,
        stop_loss=0.0,
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

            # --- Inicjalizacja DecisionEngine i RL agentów ---
            engines = {}
            rl_agents = {}
            predictions = {}
            model_outputs = {}
            agent_classes = None

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

                # RL agent (jeśli włączony)
                if rlagent and symbol in agents:
                    agent_path = os.path.join(self.AGENT_DIR, agents[symbol])
                    if os.path.exists(agent_path):
                        rl_agents[symbol] = PPO.load(agent_path, env=None)
                    else:
                        print(f"[Simulation][WARN] Brak agenta RL dla {symbol}: {agent_path}")

            # --- Symulacja ---
            num_ticks = min(len(df) for df in data.values())
            portfolio_values, timestamps, logs = [], [], []

            for t in tqdm(range(num_ticks), file=sys.stdout, dynamic_ncols=True, leave=True):
                prices = {s: data[s].iloc[t]["close"] for s in symbols}
                timestamps.append(data[symbols[0]].iloc[t]["open_time"])

                for symbol in symbols:
                    prob_array = predictions[symbol][t]
                    prob_row = dict(zip(engines[symbol].model.classes_, prob_array))

                    if rlagent and symbol in rl_agents:
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

                        decision = engines[symbol].decide(prob_row, wallet, symbol, prices[symbol])
                    else:
                        # klasyczny decision engine
                        decision = engines[symbol].decide(prob_row, wallet, symbol, prices[symbol])

                    action, qty = decision["action"], decision["qty"]

                    # Wykonanie
                    if action != "hold" and qty > 0:
                        res = wallet.execute_order(action, symbol, qty, prices[symbol])
                        executed = res["executed"]
                        action_end = res["status"] if res["status"] != "filled" else action
                    else:
                        executed, action_end = 0.0, "hold"

                    if action_end != "hold" and executed > 0:
                        logs.append({
                            "symbol": symbol,
                            "action": action_end,
                            "timestamp": data[symbol].iloc[t]["open_time"],
                            "usd_balance": f"{wallet.balance:.2f}",
                            "portfel": f"{wallet.total_value(prices):.2f}",
                            "reason": decision.get("reason", "")
                        })

                portfolio_values.append(wallet.total_value(prices))

            # --- Ocena wyników ---
            rating = Rating()
            metrics = rating.rate(portfolio_values)

            # --- Wykres ---
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, portfolio_values, label="Wartość portfela (USD)")
            plt.xlabel("Timestamp")
            plt.ylabel("Wartość portfela (USD)")
            plt.title(f"Symulacja wartości portfela ({metrics['score']})")
            plt.gca().set_xticks([])
            plt.gca().set_xticklabels([])
            plt.legend()
            plt.tight_layout()

            # --- Zapis wyników ---
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            sim_path = os.path.join(self.SIM_DIR, f"{timestamp_str}")
            os.makedirs(sim_path, exist_ok=True)

            log_path = os.path.join(sim_path, "sim_transaction_logs.txt")
            plot_path = os.path.join(sim_path, "sim_plot.png")
            results_path = os.path.join(sim_path, "sim_results.txt")

            plt.savefig(plot_path, format="png")
            plt.close()

            with open(log_path, "w", encoding="utf-8") as f:
                for line in logs:
                    f.write(json.dumps(line) + "\n")

            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            print(f"[Simulation] Zakończono w {timedelta(seconds=time.time() - start_time)}")

            return {
                "rabbit_msg_id": rabbitmq_msg_id,
                "result": "success",
                "message": "Successfully retrieved simulation data",
                "logs_file": log_path,
                "plot_file": plot_path,
                "portfolio_values": "portfolio_values",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            import traceback
            print("[Simulation][ERROR]", e)
            traceback.print_exc()
            return {
                "rabbit_msg_id": rabbitmq_msg_id,
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
