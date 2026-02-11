from .rl.base_train_service import BaseTrainService
import os
import time
from datetime import datetime, timedelta
from stable_baselines3 import PPO
import torch

class TrainAgentPPOService(BaseTrainService):
    """PPO-specific training service"""

    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        super().__init__(base_dir=base_dir, agent_type='ppo')

    def train_agent_ppo(self,
                    rabbitmq_msg_id=None,
                    model_files=None,
                    data_files=None,
                    total_timesteps=2_000_000,
                    learning_rate=0.0003,
                    n_steps=2048,
                    batch_size=64,
                    gamma=0.99,
                    ent_coef=0.01,
                    n_epochs=10,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    episode_timesteps=1024,
                    ml_threshold=0.45,
                    live_logging=True):

        try:
            start_time = time.time()
            if self.models == {}: self._load_data(model_files, data_files)

            env, callback = self._setup_env(
                total_timesteps=total_timesteps,
                base_dir=self.base_dir,
                episode_timesteps=episode_timesteps,
                live_logging=live_logging,
                ml_threshold=ml_threshold,
                num_envs=11,
                gamma=gamma
            )

            policy_kwargs = dict(
                activation_fn=torch.nn.ReLU,
                net_arch=[dict(pi=[64, 64], vf=[64, 64])]
            )

            agent = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                gamma=gamma,
                ent_coef=ent_coef,
                n_epochs=n_epochs,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                policy_kwargs=policy_kwargs,
                verbose=1
            )

            # Train
            print(f"[TrainAgentPPO] Rozpoczynanie treningu...")
            agent.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                log_interval=10000
            )

            agent_path, agent_file = self._save_agent(agent)
            env.close()


            elapsed = timedelta(seconds=time.time() - start_time)
            print(f"[TrainAgentPPO] Trening zakończony ({elapsed}).")

            return {
                "rabbit_msg_id": rabbitmq_msg_id,
                "result": "success",
                "message": "PPO training completed successfully",
                "agent_path": agent_path,
                "agent_file": agent_file,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            import traceback
            print(f"[PPO][ERROR] {e}")
            traceback.print_exc()

            return {
                "rabbit_msg_id": rabbitmq_msg_id,
                "result": "failed",
                "message": str(e),
                "agent_file": None,
                "agent_path": None,
                "timestamp": datetime.now().isoformat()
            }