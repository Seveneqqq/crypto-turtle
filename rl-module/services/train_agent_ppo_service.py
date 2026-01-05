from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env  # <--- NOWY IMPORT
from stable_baselines3.common.vec_env import SubprocVecEnv  # <--- NOWY IMPORT
from .rl.agent import TradingEnv
from .core_logic.ml_signal import ModelSignal
import os
import pandas as pd
from datetime import datetime
import torch
import multiprocessing  # <--- Do wykrycia liczby rdzeni


class TrainAgentPPOService:
    def __init__(self, base_dir=None):
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
            total_timesteps=2000000,  # Zwiększono domyślny czas, bo 8x szybciej
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            gamma=0.97,
            ent_coef=0.01):  # Zostawiam parametry z naszej dyskusji (ent_coef=0.01)

        try:
            train_params = {
                "total_timesteps": total_timesteps,
                "learning_rate": learning_rate,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "gamma": gamma,
                "ent_coef": ent_coef,
                "n_epochs": 10,
                "gae_lambda": 0.95,
                "clip_range": 0.2
            }
            print("train_params:", train_params)

            if train_params is None:
                train_params = {}
            total_timesteps = train_params.pop("total_timesteps", 10000)

            # Wczytywanie danych (To dzieje się w głównym procesie)
            models = {}
            symbols = []
            for symbol, filename in data_files.items():
                filepath = os.path.join(self.DATA_DIR, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Brak pliku: {filepath}")
                df = pd.read_csv(filepath)
                # Inicjalizacja modeli
                models[symbol] = ModelSignal(model_name=model_files[symbol], data=df.copy(), base_dir=self.base_dir)
                symbols.append(symbol)

            # # --- MULTIPROCESSING SETUP ---
            # # Ustalamy liczbę procesów równą liczbie rdzeni CPU (lub mniej, np. 8)
            # num_cpu = multiprocessing.cpu_count()
            # print(f"[TrainAgent] Uruchamianie na {num_cpu} równoległych środowiskach.")
            #
            # # Tworzymy wektor środowisk
            # # lambda: TradingEnv(...) jest potrzebna, aby każde środowisko było tworzone niezależnie w nowym procesie
            # env = make_vec_env(
            #     lambda: TradingEnv(models, symbols, base_dir=self.base_dir),
            #     n_envs=num_cpu,
            #     vec_env_cls=SubprocVecEnv,  # SubprocVecEnv = Prawdziwa równoległość (osobne procesy)
            #     wrapper_class=Monitor  # Automatycznie owija każde env w Monitor
            # )
            # # -----------------------------
            env = TradingEnv(models,symbols=symbols,base_dir=self.base_dir)
            progress_callback = ProgressBarCallback(total_timesteps=total_timesteps)
            torch.set_num_threads(16)  # Optymalizacja PyTorch pod CPU

            agent = PPO("MlpPolicy", env, verbose=1, **train_params)
            path = os.path.join(self.AGENT_DIR, "agent_ppo_2026-01-05_00-17-31.zip")
            agent = PPO.load(path, env=env, custom_objects=train_params, print_system_info=True)
            agent.learn(
                total_timesteps=total_timesteps,
                callback=progress_callback,
                log_interval=1000  # Rzadsze logowanie w konsoli SB3, bo mamy pasek
            )

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"agent_ppo_{timestamp}.zip"
            path = os.path.join(self.AGENT_DIR, filename)
            agent.save(path)

            # Ważne: Zamykamy procesy potomne
            env.close()

            return {
                "rabbit_msg_id": rabbitmq_msg_id,
                "result": "success",
                "message": "Successfully trained agent",
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
        self.progress_bar_length = 40
        self.last_progress = -1

    def _on_step(self) -> bool:
        progress = int((self.num_timesteps / self.total_timesteps) * 100)

        # Pobieranie liczby transakcji z environmentu
        trades_count = 0
        try:
            # Przy VecEnv get_attr zwraca listę wartości ze wszystkich środowisk
            trades_list = self.training_env.get_attr("trades_count")
            if trades_list:
                # Sumujemy transakcje ze wszystkich równoległych symulacji
                trades_count = sum(trades_list)
        except Exception: trades_count = "?"

        # Pobieranie Mean Reward
        mean_reward = 0.0
        if self.model and hasattr(self.model, "ep_info_buffer") and self.model.ep_info_buffer:
            mean_reward = sum([ep_info["r"] for ep_info in self.model.ep_info_buffer]) / len(self.model.ep_info_buffer)

        # Wyświetlanie (tylko jeśli postęp się zmienił, żeby nie migało za bardzo)
        if progress != self.last_progress:
            bar_filled = int(self.progress_bar_length * progress / 100)
            bar = '#' * bar_filled + '-' * (self.progress_bar_length - bar_filled)

            # Trades pokazuje teraz sumę transakcji ze wszystkich 8/16 środowisk
            sys.stdout.write(
                f"\rTraining PPO ({self.training_env.num_envs} envs): [{bar}] {progress}% | Mean Reward: {mean_reward:.4f} | Total Trades: {trades_count}")
            sys.stdout.flush()
            self.last_progress = progress

        if self.num_timesteps >= self.total_timesteps:
            print(f"\n✅ Training complete. Final Mean Reward: {mean_reward:.4f} | Total Trades: {trades_count}")

        return True