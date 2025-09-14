from typing import Any, Dict
from fastapi import APIRouter, Body
from services import rl_service

router = APIRouter(prefix="/rl", tags=["rl"])


@router.post("/anything")
async def just_send_data_to_module(req: Dict[str, Any] = Body(...)):
    rl_service.just_send_data_to_module(data=req)
    return {"status": "ok"}
