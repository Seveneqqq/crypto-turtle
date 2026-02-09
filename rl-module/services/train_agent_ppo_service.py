from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env  # <--- NOWY IMPORT
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from .rl.agent_v2 import TradingEnv
from .core_logic_v2.ml_signal import ModelSignal
from datetime import timedelta
from .rl.training_callback import TrainingCallback
from .core_logic_v2.structures import CoreConfig
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
        self.SIMDATA_DIR = os.path.join(base_dir, "shared", "files", "processed_simulation_data")
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
            ent_coef=0.01,
            episode_timesteps=1024,  # <- DODAJ TO
            ml_threshold=0.45):  # <- DODAJ TO

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

            core_config = CoreConfig(
                balance_start= 1000.0,
                symbols=symbols,
                threshold_buy=ml_threshold,
                threshold_sell=ml_threshold
            )

            env = make_vec_env(
                lambda: TradingEnv(models, num_ticks, core_config=core_config, base_dir=self.base_dir,episode_timesteps=episode_timesteps),
                n_envs=num_cpu,
                vec_env_cls=SubprocVecEnv,
                wrapper_class=Monitor
            )
            env = VecNormalize(
                env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=5.,
                clip_reward=5.,
                gamma=gamma
            )
            eval_files ={
                "BTC": "processed_ROBUST_set4_BTCUSDT-simulation_1m_2m_2025-10-11_8a076fe5-6d3c-4b1c-a490-9be3a3174517.csv"
            }
            eval_models ={}
            eval_symbols = []
            eval_num_ticks = 0
            for symbol, filename in eval_files.items():
                filepath = os.path.join(self.SIMDATA_DIR, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Brak pliku: {filepath}")
                df = pd.read_csv(filepath)
                df.dropna(inplace=True)
                len_df = len(df)
                if eval_num_ticks == 0: eval_num_ticks = len_df
                if len_df < eval_num_ticks: eval_num_ticks = len_df

                # --- Inicjalizacja modeli ---
                eval_models[symbol] = ModelSignal(model_name=model_files[symbol], data=df.copy(),
                                                  base_dir=self.base_dir)
                eval_symbols.append(symbol)

                len_pred = len(eval_models[symbol].predictions)

                if len_pred < eval_num_ticks: eval_num_ticks = len_pred

            for symbol in eval_symbols: eval_models[symbol].process_models_data(eval_num_ticks)







            # Jeden proces
            # eval_env = make_vec_env(
            #     lambda: TradingEnv(eval_models, eval_num_ticks, core_config=core_config, base_dir=self.base_dir,episode_timesteps=num_ticks),
            #     n_envs=num_cpu,
            #     vec_env_cls=SubprocVecEnv,
            #     wrapper_class=Monitor
            # )
            # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
            # eval_env.obs_rms = env.obs_rms
            # eval_callback = EvalCallback(
            #     eval_env,
            #     best_model_save_path=os.path.join(self.AGENT_DIR, "best_model"),
            #     log_path=os.path.join(self.AGENT_DIR, "eval_logs"),
            #     eval_freq=100000,  # Co ile kroków sprawdzać?
            #     deterministic=True,
            #     render=False,
            #     n_eval_episodes=1  # Jeden pełny epizod wystarczy
            # )
            # env = TradingEnv(models,symbols=symbols,base_dir=self.base_dir)

            training_callback = TrainingCallback(total_timesteps=total_timesteps, num_cpu=num_cpu, base_dir=self.base_dir, episode_timesteps=episode_timesteps)
            torch.set_num_threads(11)
            policy_kwargs = dict(
                activation_fn=torch.nn.ReLU,  # Zmiana z Tanh na ReLU (często stabilniejsze)
                net_arch=[dict(pi=[64, 64], vf=[64, 64])]  # Mniejsza sieć (domyślnie jest 64,64 ale warto sprawdzić)
            )
            agent = PPO("MlpPolicy", env, verbose=1, **train_params)

            # Douczanie agenta
            #path = os.path.join(self.AGENT_DIR, "agent_ppo_2026-01-11_23-57-11.zip")
            #agent = PPO.load(path, env=env, custom_objects=train_params, print_system_info=True)
            print(f"[TrainAgent] Rozpoczynanie treningu.")
            agent.learn(
                total_timesteps=total_timesteps,
                callback=training_callback, #, eval_callback]
                log_interval=10000
            )

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"agent_ppo_{timestamp}.zip"
            path = os.path.join(self.AGENT_DIR, filename)
            agent.save(path)

            env.close()

            print(f"[TrainAgent] Zakończono w {timedelta(seconds=time.time() - start_time)}")
            print(f"\n[TrainAgent] Training complete.")
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


