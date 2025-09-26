import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from services.rl.wallet import Wallet
from services.rl.decision import DecisionEngine
from stable_baselines3 import PPO
from services.rl.agent import TradingEnv

class SimulationService:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(base_dir, "shared", "files", "processed_files")
        self.MODEL_DIR = os.path.join(base_dir, "shared", "models", "ml")
        self.AGENT_DIR = os.path.join(base_dir, "shared", "agents")
        os.makedirs(self.AGENT_DIR, exist_ok=True)


    def run_simulation(self, rabbitmq_msg_id=None, data_file="", model_file="", initial_balance=1000.0,
                 learning_rate=3e-4, n_steps=2048, batch_size=64, total_timesteps=10_000):
        try:
            # ładujemy dane

            data_path = os.path.join(self.DATA_DIR, data_file)
            model_path = os.path.join(self.MODEL_DIR, model_file)

            df = pd.read_csv(data_path)
            df["return"] = df["close"].pct_change()
            df["ma_5"] = df["close"].rolling(window=3).mean()
            df["ma_15"] = df["close"].rolling(window=5).mean()
            df = df.dropna()

            # środowisko
            env = TradingEnv(df, model_path=model_path, initial_balance=initial_balance)

            # PPO
            model = PPO("MlpPolicy", env,
                        learning_rate=learning_rate,
                        n_steps=n_steps,
                        batch_size=batch_size,
                        verbose=1)

            model.learn(total_timesteps=total_timesteps)

            # zapisujemy agenta PPO
            save_path = os.path.join(self.AGENT_DIR, "ppo_trading_agent")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save(save_path)
            print(f"[SimulationService] Agent PPO zapisany do {save_path}")




            # Zwracamy format zgodny z gateway
            return {
                "rabbit_msg_id": rabbitmq_msg_id,
                "result": "success",
                "message": "Successfully retrieved simulation data",
                "logs_file": "log_path",
                "plot_file": "plot_path",
                "portfolio_values": "portfolio_values",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            import traceback
            print("[run_simulation][ERROR]", e)
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
