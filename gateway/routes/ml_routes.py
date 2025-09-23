# from typing import Any, Dict, List
# from fastapi import APIRouter, Body
# from pydantic import BaseModel
# from services import ml_service
#
# router = APIRouter(prefix="/ml", tags=["ml"])
#
# # 🔹 Schemat requestu (wszystko wymagane)
# class TrainRequest(BaseModel):
#     # rabbit_msg_id: str
#     df_filename: str
#     fe_version: str
#     target_version: str
#     clf_version: str
#     # feature_cols: List[str]
#     # target_col: str
#
# # 🔹 Endpoint POST
# @router.post("/train")
# async def train_model_endpoint(req: TrainRequest):
#     ml_service.train_model(
#         # rabbit_msg_id=req.rabbit_msg_id,
#         df_filename=req.df_filename,
#         fe_version=req.fe_version,
#         target_version=req.target_version,
#         clf_version=req.clf_version,
#         # feature_cols=req.feature_cols,
#         # target_col=req.target_col
#     )
#     return {"msg": "Request added to queue."}
#
# @router.post("/anything")
# async def just_send_data_to_module(req: Dict[str, Any] = Body(...)):
#     ml_service.just_send_data_to_module(data=req)
#     return {"status": "ok"}


# Zmodyfikowana wersja:


from typing import Any, Dict, List
from fastapi import APIRouter, Body
from pydantic import BaseModel
from services import ml_service

router = APIRouter(prefix="/ml", tags=["ml"])

# 🔹 Schemat requestu do trenowania
class TrainRequest(BaseModel):
    df_filename: str
    clf_version: str
    feature_cols: List[str] = None

# 🔹 Schemat requestu do przygotowania danych (dwa pliki train/test)
class PrepareFeaturesRequest(BaseModel):
    train_file: str
    test_file: str
    fe_version: str = "fe_v2"
    feature_cols: List[str] = None

# 🔹 Schemat requestu do testowania modelu
class TestModelRequest(BaseModel):
    model_filename: str
    test_filename: str

# 🔹 Endpoint POST do trenowania modelu
@router.post("/train")
async def train_model_endpoint(req: TrainRequest):
    ml_service.train_model(
        df_filename=req.df_filename,
        clf_version=req.clf_version,
        feature_cols=req.feature_cols
    )
    return {"msg": "Request added to queue."}

# 🔹 Endpoint POST do przygotowania danych
@router.post("/prepare_features")
async def prepare_features_endpoint(req: PrepareFeaturesRequest):
    ml_service.prepare_features(
        train_file=req.train_file,
        test_file=req.test_file,
        fe_version=req.fe_version,
        feature_cols=req.feature_cols
    )
    return {"msg": "Prepare features request added to queue."}

# 🔹 Endpoint POST do testowania modelu
@router.post("/test_model")
async def test_model_endpoint(req: TestModelRequest):
    ml_service.test_model(
        model_filename=req.model_filename,
        test_filename=req.test_filename
    )
    return {"msg": "Test model request added to queue."}

# 🔹 Endpoint POST do wysyłania dowolnych danych
@router.post("/anything")
async def just_send_data_to_module(req: Dict[str, Any] = Body(...)):
    ml_service.just_send_data_to_module(data=req)
    return {"status": "ok"}
