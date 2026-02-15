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

def block_train_sac():
    TrainAgentSACService = load(
        "rl-module.services.train_agent_sac_service",
        "TrainAgentSACService"
    )
    service = TrainAgentSACService(base_dir)
    return service

def block_ppo_tuner():
    PPOTunerService = load(
        "rl-module.services.train_agent_ppo_tuner_service",
        "PPOTunerService"
    )
    service = PPOTunerService(base_dir)
    return service

def block_sac_tuner():
    SACTunerService = load(
        "rl-module.services.train_agent_sac_tuner_service",
        "SACTunerService"
    )
    service = SACTunerService(base_dir)
    return service

def block_dqn_tuner():
    DQNTunerService = load(
        "rl-module.services.train_agent_dqn_tuner_service",
        "DQNTunerService"
    )
    service = DQNTunerService(base_dir)
    return service

def block_simulation_tuner():
    SimulationTunerService = load(
        "rl-module.services.simulation_tuner_service",
        "SimulationTunerService"
    )
    service = SimulationTunerService(base_dir)
    return service

def block_simulation():
    SimulationService = load(
        "test-module.services.simulation_service",
        "SimulationService"
    )
    service = SimulationService(base_dir)
    return service

def block_visualisation():
    pass


