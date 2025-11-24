from typing import Any, Dict
from fastapi import APIRouter, Body
from services import rl_service
from pydantic import BaseModel
router = APIRouter(prefix="/rl", tags=["rl"])

class TrainAgentPPORequest(BaseModel):
    model_files: Dict[str, str]
    data_files: Dict[str, str]
    total_timesteps: int
    learning_rate: float
    n_steps: int
    batch_size: int
    gamma: float
    ent_coef: float

class TrainAgentDQNRequest(BaseModel):
    model_file: str
    data_file: str
    total_timesteps: int
    learning_rate: float
    n_steps: int
    batch_size: int
    gamma: float
    ent_coef: float

@router.post("/train_agent_ppo")
def train_agent_ppo(req: TrainAgentPPORequest):
    rl_service.train_agent_ppo(
        model_files=req.model_files,
        data_files=req.data_files,
        total_timesteps=req.total_timesteps,
        learning_rate=req.learning_rate,
        n_steps=req.n_steps,
        batch_size=req.batch_size,
        gamma=req.gamma,
        ent_coef=req.ent_coef,
    )
    return {"msg": "TrainAgentPPORequest added to queue."}

@router.post("/train_agent_dqn")
def train_agent_dqn(req: TrainAgentDQNRequest):
    rl_service.train_agent_dqn(
        model_file=req.model_file,
        data_file=req.data_file,
        total_timesteps=req.total_timesteps,
        learning_rate=req.learning_rate,
        n_steps=req.n_steps,
        batch_size=req.batch_size,
        gamma=req.gamma,
        ent_coef=req.ent_coef,
    )
    return {"msg": "TrainAgentDQNRequest added to queue."}

@router.post("/anything")
async def just_send_data_to_module(req: Dict[str, Any] = Body(...)):
    rl_service.just_send_data_to_module(data=req)
    return {"status": "ok"}
