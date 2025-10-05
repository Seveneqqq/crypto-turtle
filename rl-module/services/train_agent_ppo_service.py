from stable_baselines3 import PPO
from services.rl.agent import TradingEnv
import os
import pandas as pd
import joblib
from datetime import datetime


class TrainAgentPPOService:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(base_dir, "shared", "files", "processed_rlagent_data")
        self.MODEL_DIR = os.path.join(base_dir, "shared", "models", "sklearn")
        self.AGENT_DIR = os.path.join(base_dir, "shared", "agents", "ppo")
        os.makedirs(self.AGENT_DIR, exist_ok=True)


    def train_agent_ppo(
            self,
            rabbitmq_msg_id=None,
            model_file=None,
            data_file=None,
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
                "ent_coef": ent_coef
            }
            print("train_params:", train_params)

            if train_params is None:
                train_params = {}
            total_timesteps = train_params.pop("total_timesteps", 10000)

            data_path = os.path.join(self.DATA_DIR, data_file)
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Brak pliku: {data_file}")

            data = pd.read_csv(data_path)
            model_path = os.path.join(self.MODEL_DIR, model_file)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Brak pliku: {model_file}")
            model_ml = joblib.load(model_path)
            env = TradingEnv(data, model_ml)
            agent = PPO("MlpPolicy", env, verbose=1, **train_params)
            agent.learn(total_timesteps=total_timesteps)

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