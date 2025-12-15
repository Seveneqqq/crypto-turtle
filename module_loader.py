import importlib
import os

os.environ["SKIP_INIT"] = "1"
base_dir = os.path.dirname(os.path.abspath(__file__))
def load(name: str, attr: str):
    module = importlib.import_module(name)
    return getattr(module, attr)

def block_train_ppo():
    TrainAgentPPOService = load(
        "rl-module.services.train_agent_ppo_service",
        "TrainAgentPPOService"
    )
    service = TrainAgentPPOService(base_dir)
    return service

def block_simulation():
    pass

def block_visualisation():
    pass


