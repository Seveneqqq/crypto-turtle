import os
import pandas as pd
import multiprocessing
from datetime import datetime
from abc import ABC
import torch
from plotly.data import experiment
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from ..core_logic_v2.ml_signal import ModelSignal
from ..core_logic_v2.structures import CoreConfig
from .agent_v2 import TradingEnv
from .training_callback import TrainingCallback

class BaseTrainService(ABC):

    def __init__(self, base_dir=None, agent_type='ppo'):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.base_dir = base_dir
        self.agent_type = agent_type

        self.DATA_DIR = os.path.join(base_dir, "shared", "files", "processed_rlagent_data")
        self.AGENT_DIR = os.path.join(base_dir, "shared", "agents", agent_type)
        os.makedirs(self.AGENT_DIR, exist_ok=True)

        self.models = {}
        self.symbols = []
        self.num_ticks = 0


    def _load_data(self, model_files, data_files):
        """ Funkcja ładująca dane."""

        print(f"[TrainAgent{self.agent_type.upper()}] Przygotowywanie modeli i danych....")

        for symbol, filename in data_files.items():
            filepath = os.path.join(self.DATA_DIR, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Brak pliku: {filepath}")

            df = pd.read_csv(filepath)
            df.dropna(inplace=True)

            len_df = len(df)
            if self.num_ticks == 0:
                self.num_ticks = len_df
            else:
                self.num_ticks = min(self.num_ticks, len_df)

            self.models[symbol] = ModelSignal(
                model_name=model_files[symbol],
                data=df.copy(),
                base_dir=self.base_dir
            )
            self.symbols.append(symbol)

            len_pred = len(self.models[symbol].predictions)
            self.num_ticks = min(self.num_ticks, len_pred)

        for symbol in self.symbols:
            self.models[symbol].process_models_data(self.num_ticks)

        print(f"[TrainAgent{self.agent_type.upper()}] Załadowano {len(self.symbols)} kryptowalut, {self.num_ticks} timestamps.")
        return True

    def _setup_env(self,
                   total_timesteps,
                   base_dir,
                   live_logging,
                   episode_timesteps,
                   ml_threshold,
                   num_envs=None,
                   normalize=True,
                   gamma=0.99,
                   experiment_version="v0"):
        """Przygotowanie środowiska tradingowego"""

        if num_envs is None:
            num_envs = max(1, multiprocessing.cpu_count() - 5)

        print(f"[TrainAgent{self.agent_type.upper()}] Uruchamianie na {num_envs} równoległych środowiskach...")

        core_config = CoreConfig(
            balance_start=1000.0,
            symbols=self.symbols,
            threshold_buy=ml_threshold,
            threshold_sell=ml_threshold,
            experiment_version=experiment_version,
        )

        env = make_vec_env(
            lambda: TradingEnv(
                models=self.models,
                num_ticks=self.num_ticks,
                base_dir=self.base_dir,
                core_config=core_config,
                episode_timesteps=episode_timesteps,
                agent_type=self.agent_type
            ),
            n_envs=num_envs,
            vec_env_cls=SubprocVecEnv,
            wrapper_class=Monitor
        )
        torch.set_num_threads(num_envs)

        if normalize:
            env = VecNormalize(
                env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=5.0,
                clip_reward=5.0,
                gamma=gamma
            )
        #TODO:
        # callbacks.append(
        #     CheckpointCallback(
        #         save_freq=checkpoint_freq,
        #         save_path=os.path.dirname(save_path),
        #         name_prefix="sac_trading"
        #     )
        # )
        callback = TrainingCallback(
            total_timesteps=total_timesteps,
            num_cpu=env.num_envs,
            base_dir=base_dir,
            episode_timesteps=episode_timesteps,
            live=live_logging
        )
        return env, callback

    def _save_agent(self, agent):
        """Zapis agenta."""

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"agent_{self.agent_type}_{timestamp}.zip"
        path = os.path.join(self.AGENT_DIR, filename)

        agent.save(path)
        print(f"TrainAgent[{self.agent_type.upper()}] Agent zapisany do: {path}")

        return path, filename