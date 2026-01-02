from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
class SACAgent:
    def __init__(
        self,
        env,
        model_path: str | None = None,
        device: str = "auto",
        verbose: int = 1
    ):
        """
        env: TradingEnv (ten sam co PPO)
        model_path: ścieżka do .zip (jeśli inference)
        """



        self.env = DummyVecEnv([lambda: env])
        self.env = VecNormalize(
            self.env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=5.0,
            clip_reward=5.0
        )

        if model_path and os.path.exists(model_path):
            self.model = SAC.load(
                model_path,
                env=self.env,
                device=device
            )
        else:
            self.model = SAC(
                policy="MlpPolicy",
                env=self.env,

                # 🔑 learning
                learning_rate=1e-4,  # stabilne dla SAC
                gamma=0.98,  # długoterminowy reward (trading!)
                tau=0.005,

                # 🧠 replay buffer
                buffer_size=100_000,  # ~2x długość danych
                learning_starts=10_000,  # NIE ucz się od razu

                # 🔄 aktualizacje
                train_freq=1,  # update co step
                gradient_steps=1,  # 1 update = stabilność
                batch_size=256,  # sweet spot CPU

                # 🎯 entropy (ważne!)
                ent_coef="auto",  # KLUCZOWE: ogranicza eksplozję actora

                # 🧠 sieć
                policy_kwargs=dict(
                    net_arch=[256, 256],
                    activation_fn=torch.nn.ReLU
                ),

                # 🔍 target
                target_update_interval=1,

                verbose=1,
                device=device
            )

    def train(
        self,
        total_timesteps: int,
        save_path: str | None = None,
        checkpoint_freq: int = 200_000
    ):
        callbacks = []

        if save_path:
            callbacks.append(
                CheckpointCallback(
                    save_freq=checkpoint_freq,
                    save_path=os.path.dirname(save_path),
                    name_prefix="sac_trading"
                )
            )

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks
        )

        if save_path:
            self.model.save(save_path)

    def predict(self, obs):
        """
        obs: observation z TradingEnv
        """
        action, _ = self.model.predict(
            obs,
            deterministic=False
        )
        return action

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = SAC.load(path, env=self.env)
