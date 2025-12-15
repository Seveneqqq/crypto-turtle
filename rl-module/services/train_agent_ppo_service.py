from stable_baselines3 import PPO
from .rl.agent import TradingEnv
from .rl.rl_ml_signal import ModelSignal
import os
import pandas as pd
from datetime import datetime
import torch


class TrainAgentPPOService:
    def __init__(self, base_dir = None):
        if base_dir is None: base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.DATA_DIR = os.path.join(base_dir, "shared", "files", "processed_rlagent_data")
        self.MODEL_DIR = os.path.join(base_dir, "shared", "models", "sklearn")
        self.AGENT_DIR = os.path.join(base_dir, "shared", "agents", "ppo")
        os.makedirs(self.AGENT_DIR, exist_ok=True)


    def train_agent_ppo(
            self,
            rabbitmq_msg_id=None,
            model_files=None,
            data_files=None,
            total_timesteps= 10000,
            learning_rate= 0.0003,
            n_steps= 2048,
            batch_size= 64,
            gamma= 0.99,
            ent_coef= 0.0):


        try:
            train_params = {
                "total_timesteps": total_timesteps,
                "learning_rate": learning_rate,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "gamma": gamma,
                "ent_coef": ent_coef,
                "n_epochs": 8,
                "gae_lambda": 0.95,
                "clip_range": 0.2
            }
            print("train_params:", train_params)

            if train_params is None:
                train_params = {}
            total_timesteps = train_params.pop("total_timesteps", 10000)

            models = {}
            symbols = []
            for symbol, filename in data_files.items():
                filepath = os.path.join(self.DATA_DIR, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Brak pliku: {filepath}")
                df = pd.read_csv(filepath)
                # --- Inicjalizacja modeli ---
                models[symbol] = ModelSignal(model_name=model_files[symbol], data=df.copy(), base_dir=self.base_dir)
                symbols.append(symbol)

            env = TradingEnv(models, symbols)

            progress_callback = ProgressBarCallback(total_timesteps=total_timesteps)
            torch.set_num_threads(16)
            agent = PPO("MlpPolicy", env, verbose=1, **train_params)
            agent.learn(
                total_timesteps=total_timesteps,
                callback=progress_callback,
                log_interval=10000
            )

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"agent_ppo_{timestamp}.zip"
            path = os.path.join(self.AGENT_DIR, filename)
            agent.save(path)

            return {
                "rabbit_msg_id": rabbitmq_msg_id,
                "result": "success",
                "message": "Successfully retrieved simulation data",
                "agent_file": path,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            import traceback
            print("[TrainAgent][ERROR]", e)
            traceback.print_exc()
            return {
                "rabbit_msg_id": rabbitmq_msg_id,
                "result": "failed",
                "message": str(e),
                "agent_file": None,
                "timestamp": datetime.now().isoformat()
            }

import sys
from stable_baselines3.common.callbacks import BaseCallback

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.progress_bar_length = 40  # ile znaków w pasku
        self.last_progress = -1

    def _on_step(self) -> bool:
        progress = int((self.num_timesteps / self.total_timesteps) * 100)

        # aktualizuj tylko jeśli się zmienił
        if progress != self.last_progress:
            bar_filled = int(self.progress_bar_length * progress / 100)
            bar = '#' * bar_filled + '-' * (self.progress_bar_length - bar_filled)
            sys.stdout.write(f"\rTraining PPO agent: [{bar}] {progress}%")

            sys.stdout.flush()
            self.last_progress = progress

        # zakończenie
        if self.num_timesteps >= self.total_timesteps:
            print("\n✅ Training complete.")

        return True