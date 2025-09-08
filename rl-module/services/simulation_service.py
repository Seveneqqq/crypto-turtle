import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from services.rl.wallet import Wallet
from services.rl.decision import DecisionEngine

class SimulationService:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(base_dir, "shared", "files", "processed_files")
        self.MODEL_DIR = os.path.join(base_dir, "shared", "models", "ml")
        self.LOGS_DIR = os.path.join(base_dir, "shared", "simulation", "logs")
        self.PLOTS_DIR = os.path.join(base_dir, "shared", "simulation", "plots")

    def run_simulation(self, rabbitmq_msg_id=None, balance_start=1000.0, time_step=1,
                       data_files=None, models=None, threshold_buy=0.3, threshold_sell=0.7):
        try:
            print("run_simulation")
            if data_files is None:
                data_files = {}
            if models is None:
                models = {}

            # Ładowanie danych
            data = {}
            for symbol, filename in data_files.items():
                filepath = os.path.join(self.DATA_DIR, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Brak pliku: {filepath}")
                df = pd.read_csv(filepath)
                data[symbol] = df[["timestamp", "close"]].copy()

            symbols = list(data.keys())

            # Inicjalizacja portfela i engine
            wallet = Wallet(balance_start, symbols)
            engines = {}
            for symbol in symbols:
                model_path = os.path.join(self.MODEL_DIR, models[symbol])
                engines[symbol] = DecisionEngine(model_path=model_path, threshold=threshold_buy)

            num_ticks = min(len(df) for df in data.values())
            portfolio_values = []
            timestamps = []
            logs = []

            for t in range(num_ticks):
                prices = {symbol: data[symbol].iloc[t]["close"] for symbol in symbols}
                timestamps.append(data[symbols[0]].iloc[t]["timestamp"])

                for symbol in symbols:
                    history = data[symbol].iloc[:t + 1].copy()
                    if len(history) < 15:
                        continue
                    features = self.extract_features(history)
                    amount, action = engines[symbol].decide(features)
                    action_end, amount_end = wallet.execute(action, symbol, prices[symbol], amount)

                    if action_end != "hold":
                        logs.append({
                            "symbol": symbol,
                            "action": action_end,
                            "timestamp": history.iloc[-1]["timestamp"],
                            "usd_balance": f"{wallet.balance:.2f}",
                            "portfel": f"{wallet.total_value(prices):.2f}"
                        })

                portfolio_values.append(wallet.total_value(prices))

            # Wykres
            plt.figure(figsize=(12,6))
            plt.plot(timestamps, portfolio_values, label="Wartość portfela (USD)")
            plt.xlabel("Timestamp")
            plt.ylabel("Wartość portfela (USD)")
            plt.title("Symulacja wartości portfela")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(self.LOGS_DIR, f"{timestamp_str}_simulation_logs.txt")
            plot_path = os.path.join(self.PLOTS_DIR, f"{timestamp_str}_simulation_plot.png")
            plt.savefig(plot_path, format="png")
            plt.close()

            # Zapis logów
            with open(log_path, "w", encoding="utf-8") as f:
                for line in logs:
                    f.write(json.dumps(line) + "\n")

            # Zwracamy format zgodny z gateway
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
            return {
                "rabbit_msg_id": rabbitmq_msg_id,
                "result": "failed",
                "message": str(e),
                "logs_file": None,
                "plot_file": None,
                "portfolio_values": [],
                "timestamp": datetime.now().isoformat()
            }

    @staticmethod
    def extract_features(df):
        df = df.copy()
        df["return"] = df["close"].pct_change()
        df["ma_5"] = df["close"].rolling(window=3).mean()
        df["ma_15"] = df["close"].rolling(window=5).mean()
        df = df.dropna()
        if df.empty:
            return None
        features = ["close", "return", "ma_5", "ma_15"]
        return df[features].iloc[[-1]]
