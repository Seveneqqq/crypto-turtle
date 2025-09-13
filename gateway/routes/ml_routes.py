from typing import List
from fastapi import APIRouter
from pydantic import BaseModel
from services import ml_service

router = APIRouter(prefix="/ml", tags=["ml"])

# 🔹 Schemat requestu (wszystko wymagane)
class TrainRequest(BaseModel):
    # rabbit_msg_id: str
    df_filename: str
    fe_version: str
    target_version: str
    clf_version: str
    # feature_cols: List[str]
    # target_col: str

# 🔹 Endpoint POST
@router.post("/train")
async def train_model_endpoint(req: TrainRequest):
    ml_service.train_model(
        # rabbit_msg_id=req.rabbit_msg_id,
        df_filename=req.df_filename,
        fe_version=req.fe_version,
        target_version=req.target_version,
        clf_version=req.clf_version,
        # feature_cols=req.feature_cols,
        # target_col=req.target_col
    )
    return {"msg": "Request added to queue."}
