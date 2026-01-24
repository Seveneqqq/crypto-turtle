import csv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env  # <--- NOWY IMPORT
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from .rl.agent import TradingEnv
from .core_logic.ml_signal import ModelSignal
from datetime import timedelta
from .rl.training_callback import TrainingCallback
import time
import os
import pandas as pd
from datetime import datetime
import torch
import multiprocessing

class TrainAgentPPOService:
    def __init__(self, base_dir=None):
        if base_dir is None: base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.DATA_DIR = os.path.join(base_dir, "shared", "files", "processed_rlagent_data")
        self.AGENT_DIR = os.path.join(base_dir, "shared", "agents", "ppo")
        os.makedirs(self.AGENT_DIR, exist_ok=True)

    def train_agent_ppo(
            self,
            rabbitmq_msg_id=None,
            model_files=None,
            data_files=None,
            total_timesteps=2000000,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            gamma=0.97,
            ent_coef=0.01):

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
            start_time = time.time()

            #TODO: poniżej

            episode_timesteps = 10240



            if train_params is None:
                train_params = {}
            total_timesteps = train_params.pop("total_timesteps", 10000)

            models = {}
            symbols = []
            num_ticks = 0
            print(f"[TrainAgent] Przygotowywanie modeli i danych.")
            # Dodać sprawdzanie open_time
            for symbol, filename in data_files.items():
                filepath = os.path.join(self.DATA_DIR, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Brak pliku: {filepath}")
                df = pd.read_csv(filepath)
                df.dropna(inplace=True)
                len_df = len(df)
                if num_ticks == 0: num_ticks = len_df
                if len_df < num_ticks: num_ticks = len_df

                # --- Inicjalizacja modeli ---
                models[symbol] = ModelSignal(model_name=model_files[symbol], data=df.copy(),
                                                  base_dir=self.base_dir)
                symbols.append(symbol)

                len_pred = len(models[symbol].predictions)

                if len_pred < num_ticks: num_ticks = len_pred

            for symbol in symbols: models[symbol].process_models_data(num_ticks)

            num_cpu = multiprocessing.cpu_count() - 5
            print(f"[TrainAgent] Uruchamianie na {num_cpu} równoległych środowiskach.")

            env = make_vec_env(
                lambda: TradingEnv(models, symbols,num_ticks, base_dir=self.base_dir),
                n_envs=num_cpu,
                vec_env_cls=SubprocVecEnv,
                wrapper_class=Monitor
            )
            env = VecNormalize(
                env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.,
                clip_reward=10.,
                gamma=gamma
            )

            # Jeden proces

            # env = TradingEnv(models,symbols=symbols,base_dir=self.base_dir)

            training_callback = TrainingCallback(total_timesteps=total_timesteps, num_cpu=num_cpu, base_dir=self.base_dir, episode_timesteps=episode_timesteps)
            torch.set_num_threads(11)

            agent = PPO("MlpPolicy", env, verbose=1, **train_params)

            # Douczanie agenta
            #path = os.path.join(self.AGENT_DIR, "agent_ppo_2026-01-11_23-57-11.zip")
            #agent = PPO.load(path, env=env, custom_objects=train_params, print_system_info=True)
            print(f"[TrainAgent] Rozpoczynanie treningu.")
            agent.learn(
                total_timesteps=total_timesteps,
                callback=training_callback,
                log_interval=10000
            )

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"agent_ppo_{timestamp}.zip"
            path = os.path.join(self.AGENT_DIR, filename)
            agent.save(path)

            env.close()
            print(f"\n[TrainAgent] Training complete.")
            print(f"[TrainAgent] Zakończono w {timedelta(seconds=time.time() - start_time)}")
            return {
                "rabbit_msg_id": rabbitmq_msg_id,
                "result": "success",
                "message": "Successfully trained agent",
                "agent_path": path,
                "agent_file": filename,
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


