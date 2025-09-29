import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

from services.test.wallet import Wallet, WalletConfig
from services.test.decision import DecisionEngine, DecisionConfig
from services.test.rating import Rating

class SimulationService:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(base_dir, "shared", "files", "processed_files")
        self.MODEL_DIR = os.path.join(base_dir, "shared", "models", "ml")
        self.SIM_DIR = os.path.join(base_dir, "shared", "simulation")

        # Domyślne wartości parametrów decyzyjnych (mogą być nadpisane w run_simulation)
        self.wallet_percentage_limit = 0.5
        self.mincost = 1.0
        self.stop_loss = 1.0
        self.fee = 0.0
        self.slippage = 0
        self.max_wallet_percentage_crypto = 1.0
        self.rlagent = False  # czy RL steruje decyzjami (na razie False)

    def run_simulation(
        self,
        rabbitmq_msg_id=None,
        balance_start=1000.0,
        time_step=1,
        data_files=None,
        models=None,
        threshold_buy=0.6,
        threshold_sell=0.6,
        wallet_percentage_limit=None,
        mincost=None,
        stop_loss=None,
        fee=None,
        slippage=None,
        max_wallet_percentage_crypto=None
    ):
        try:
            print("[Simulation] Start")
            start_time = time.time()

            # Parametry decyzyjne nadpisane jeśli podano
            wallet_percentage_limit = wallet_percentage_limit or self.wallet_percentage_limit
            mincost = mincost or self.mincost
            stop_loss = stop_loss or self.stop_loss
            fee = fee or self.fee
            slippage = slippage or self.slippage
            max_wallet_percentage_crypto = max_wallet_percentage_crypto or self.max_wallet_percentage_crypto

            if data_files is None:
                data_files = {}
            if models is None:
                models = {}

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
            config = WalletConfig(
                wallet_percentage_limit=0.9,
                mincost=1.0,
                slippage=0.0,
                stop_loss=0.05,
                fee=0.001,
                max_wallet_percentage_crypto=0.5
            )
            wallet = Wallet(balance_start, symbols, config)

            # --- Inicjalizacja DecisionEngine i predykcji ---
            config = DecisionConfig(
                threshold_buy=threshold_buy,
                threshold_sell=threshold_sell,
                tx_fraction=0.01,
                max_fraction_when_confident=0.5,
                mincost=1.0
            )

            engines = {}
            predictions = {}
            for symbol in symbols:
                model_path = os.path.join(self.MODEL_DIR, models[symbol])
                engines[symbol] = DecisionEngine(model_path=model_path, config=config)

                history = data[symbol].copy()
                features = engines[symbol].model.feature_names_in_
                x = history[features].values
                probs = engines[symbol].model.predict_proba(x)
                predictions[symbol] = probs

            num_ticks = min(len(df) for df in data.values())
            portfolio_values = []
            timestamps = []
            logs = []

            # --- Symulacja pętli czasowej ---
            for t in range(num_ticks):
                prices = {symbol: data[symbol].iloc[t]["close"] for symbol in symbols}
                timestamps.append(data[symbols[0]].iloc[t]["open_time"])

                for symbol in symbols:
                    prob_array = predictions[symbol][t]
                    prob_row = dict(zip(engines[symbol].model.classes_, prob_array))

                    # Decision
                    decision = engines[symbol].decide(prob_row, wallet, symbol, prices[symbol])
                    action = decision["action"]
                    qty = decision["qty"]

                    # Execute
                    if action != "hold" and qty > 0:
                        res = wallet.execute_order(action, symbol, qty, prices[symbol])
                        executed = res["executed"]
                        action_end = res["status"] if res["status"] != "filled" else action
                    else:
                        executed = 0.0
                        action_end = "hold"

                    # Log if something happened
                    if action_end != "hold" and executed > 0:
                        logs.append({
                            "symbol": symbol,
                            "action": action_end,
                            "timestamp": data[symbol].iloc[t]["open_time"],
                            "usd_balance": f"{wallet.balance:.2f}",
                            "portfel": f"{wallet.total_value(prices):.2f}",
                            "reason": decision.get("reason", "")
                        })

                if t in (0, num_ticks - 1) or (t % 1000 == 0):
                    print(f"[Simulation] Progress = {(t / num_ticks) * 100:.2f}%")

                portfolio_values.append(wallet.total_value(prices))

            # --- Ocena wyników ---
            rating = Rating()
            metrics = rating.rate(portfolio_values)

            # --- Wykres ---
            plt.figure(figsize=(12,6))
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

            # Zapis wykresu
            plt.savefig(plot_path, format="png")
            plt.close()

            # Zapis logów
            with open(log_path, "w", encoding="utf-8") as f:
                for line in logs:
                    f.write(json.dumps(line) + "\n")

            # Zapis wyników JSON
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            end_time = time.time()
            duration = timedelta(seconds=end_time - start_time)
            print(f"[Simulation] Zakończono w {duration}")

            return {
                "rabbit_msg_id": rabbitmq_msg_id,
                "result": "success",
                "message": "Successfully retrieved simulation data",
                "logs_file": log_path,
                "plot_file": plot_path,
                "portfolio_values": portfolio_values,
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



# import os
# import json
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from datetime import datetime
# import time
# from datetime import timedelta
# from services.test.wallet import Wallet
# from services.test.decision import DecisionEngine
# from services.test.rating import Rating
#
# class SimulationService:
#     def __init__(self):
#         base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         self.DATA_DIR = os.path.join(base_dir, "shared", "files", "processed_files")
#         self.MODEL_DIR = os.path.join(base_dir, "shared", "models", "ml")
#         self.SIM_DIR = os.path.join(base_dir, "shared", "simulation")
#
#
#         # nie dodane wartości do gateway
#
#         self.wallet_percentage_limit = 0.9  # procent (0.0-1.0) maksymalnej wartości transakcji dla portfela
#         self.mincost = 1.0                  # minimalna wartość transakcji w USDT
#         self.stop_loss = 1.0               # stop_loss dla wartości posiadanej kryptowaluty
#         self.fee = 0.0                      # prowizja przy transakcji
#
#         self.rlagent = False                # symulacja z rlagentem lub bez --brak
#         self.slippage = 0                   # opóźnienie w kupnie/sprzedaży w tickach --brak
#         self.max_wallet_percentage_crypto = 1.0  #dywesyfikacja portfela --brak
#
#     def run_simulation(self, rabbitmq_msg_id=None, balance_start=1000.0, time_step=1,
#                        data_files=None, models=None, threshold_buy=0.6, threshold_sell=0.6):
#         try:
#             print("Start Simulation")
#             start = time.time()
#
#             if data_files is None:
#                 data_files = {}
#             if models is None:
#                 models = {}
#
# # Ładowanie danych
#
#             data = {}
#
#             for symbol, filename in data_files.items():
#                 filepath = os.path.join(self.DATA_DIR, filename)
#                 if not os.path.exists(filepath):
#                     raise FileNotFoundError(f"Brak pliku: {filepath}")
#                 df = pd.read_csv(filepath)
#                 data[symbol] = df.copy()
#
#             symbols = list(data.keys())
#
# # Inicjalizacja portfela i engine
#
#             wallet = Wallet(balance_start, symbols,
#                             threshold_buy=threshold_buy,
#                             threshold_sell=threshold_sell,
#                             wallet_percentage_limit=self.wallet_percentage_limit,
#                             mincost=self.mincost,
#                             slippage=self.slippage,
#                             stop_loss=self.stop_loss,
#                             fee=self.fee,
#                             max_wallet_percentage_crypto=self.max_wallet_percentage_crypto)
#             engines = {}
#             predictions = {}
#             for symbol in symbols:
#                 model_path = os.path.join(self.MODEL_DIR, models[symbol])
#                 engines[symbol] = DecisionEngine(model_path=model_path, threshold_buy=threshold_buy,threshold_sell=threshold_sell)
#
# # Przewidywanie modelu dla wszystkich wartości
#
#                 history = data[symbol].copy()
#                 features = engines[symbol].model.feature_names_in_
#                 x = history[features].values
#                 probs = engines[symbol].model.predict_proba(x)
#                 predictions[symbol] = probs
#
#             num_ticks = min(len(df) for df in data.values())
#             portfolio_values = []
#             timestamps = []
#             logs = []
#
#             for t in range(0, num_ticks):
#                 prices = {symbol: data[symbol].iloc[t]["close"] for symbol in symbols}
#                 timestamps.append(data[symbols[0]].iloc[t]["open_time"])
#
#                 for symbol in symbols:
#
#                     probs_array = predictions[symbol][t]
#                     prob_row = dict(zip(engines[symbol].model.classes_, probs_array))
#                     amount, action = engines[symbol].decide(prob_row)
#
#                     if action != "hold":
#                         action_end, amount_end = wallet.execute(action, symbol, prices[symbol], amount)
#                     else:
#                         action_end = action
#
#                     if action_end != "hold":
#                         logs.append({
#                             "symbol": symbol,
#                             "action": action_end,
#                             "timestamp": data[symbol].iloc[t]["open_time"],
#                             "usd_balance": f"{wallet.balance:.2f}",
#                             "portfel": f"{wallet.total_value(prices):.2f}"
#                         })
#
#                 if t in (0, num_ticks - 1) or (t % 1000 == 0):
#                     print(f"Progress = {(t / num_ticks) * 100:.2f}%")
#                 portfolio_values.append(wallet.total_value(prices))
#
#             rating = Rating()
#             metrics = rating.rate(portfolio_values)
#             # Wykres
#             plt.figure(figsize=(12,6))
#             plt.plot(timestamps, portfolio_values, label="Wartość portfela (USD)")
#             plt.xlabel("Timestamp")
#             plt.ylabel("Wartość portfela (USD)")
#             plt.title(f"Symulacja wartości portfela ({metrics["score"]})")
#             plt.gca().set_xticklabels([])  # usuwa napisy
#             plt.gca().set_xticks([])  # usuwa też kreski
#             plt.legend()
#             plt.tight_layout()
#
# # Zapis plików z wynikami do shared/simulation
#
#             timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#             sim_path = os.path.join(self.SIM_DIR, f"{timestamp_str}")
#             os.makedirs(sim_path, exist_ok=True)
#             log_path = os.path.join(sim_path, "sim_transaction_logs.txt")
#             plot_path = os.path.join(sim_path, "sim_plot.png")
#             results_path = os.path.join(sim_path, "sim_results.txt")
#
#             # Zapis wykresu
#             print(f"Zapis wykresu do: {plot_path}")
#             plt.savefig(plot_path, format="png")
#             plt.close()
#
#             # Zapis logów
#             print(f"Zapis logów do: {log_path}")
#             with open(log_path, "w", encoding="utf-8") as f:
#                 for line in logs:
#                     f.write(json.dumps(line) + "\n")
#
#
#
#
#             # zapis JSON
#             with open(results_path, "w", encoding="utf-8") as f:
#                 json.dump(metrics, f, indent=2)
#
#
#             # Zapis wyników
#             end = time.time()
#             duration = timedelta(seconds=end - start)
#             print(f"Zapis wyników do: {results_path}")
#             # with open(results_path, "w", encoding="utf-8") as f:
#             #         f.write(json.dumps("tu będą wyniki, dane o symulacji itp.") + "\n")
#             #         f.write(f"Czas trwania symulacji: {duration}\n")
#
#             # Zwracamy format zgodny z gateway
#             return {
#                 "rabbit_msg_id": rabbitmq_msg_id,
#                 "result": "success",
#                 "message": "Successfully retrieved simulation data",
#                 "logs_file": log_path,
#                 "plot_file": plot_path,
#                 "portfolio_values": "portfolio_values",
#                 "timestamp": datetime.now().isoformat()
#             }
#
#         except Exception as e:
#             import traceback
#             print("[run_simulation][ERROR]", e)
#             traceback.print_exc()
#             return {
#                 "rabbit_msg_id": rabbitmq_msg_id,
#                 "result": "failed",
#                 "message": str(e),
#                 "logs_file": None,
#                 "plot_file": None,
#                 "portfolio_values": [],
#                 "timestamp": datetime.now().isoformat()
#             }
