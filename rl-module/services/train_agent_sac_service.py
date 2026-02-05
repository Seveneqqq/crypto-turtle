from .rl.agent_sac import SACAgent
from .rl.agent import TradingEnv
from .core_logic.ml_signal import ModelSignal
import os
import pandas as pd
from datetime import datetime
import torch

class TrainAgentSACService:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.DATA_DIR = os.path.join(base_dir, "shared", "files", "processed_rlagent_data")
        self.MODEL_DIR = os.path.join(base_dir, "shared", "models", "sklearn")
        self.AGENT_DIR = os.path.join(base_dir, "shared", "agents", "sac")
        os.makedirs(self.AGENT_DIR, exist_ok=True)

    def train_agent_sac(
        self,
        rabbitmq_msg_id=None,
        model_files=None,
        data_files=None,
        total_timesteps=20000,
        checkpoint_freq=200_000,
        learning_rate=3e-4,
        device="auto"
    ):
        try:
            # --- Przygotowanie danych ---
            models = {}
            symbols = []
            for symbol, filename in data_files.items():
                filepath = os.path.join(self.DATA_DIR, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Brak pliku: {filepath}")
                df = pd.read_csv(filepath)
                models[symbol] = ModelSignal(model_name=model_files[symbol], data=df.copy(), base_dir=self.base_dir)
                symbols.append(symbol)

            env = TradingEnv(models, symbols, base_dir=self.base_dir)

            # --- Uruchomienie agenta SAC ---
            torch.set_num_threads(16)
            agent = SACAgent(env=env, device=device)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"agent_sac_{timestamp}.zip"
            save_path = os.path.join(self.AGENT_DIR, filename)

            agent.train(
                total_timesteps=total_timesteps,
                save_path=save_path,
                checkpoint_freq=checkpoint_freq
            )

            return {
                "rabbit_msg_id": rabbitmq_msg_id,
                "result": "success",
                "message": "SAC training completed",
                "agent_file": save_path,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            import traceback
            print("[TrainAgentSAC][ERROR]", e)
            traceback.print_exc()
            return {
                "rabbit_msg_id": rabbitmq_msg_id,
                "result": "failed",
                "message": str(e),
                "agent_file": None,
                "timestamp": datetime.now().isoformat()
            }
