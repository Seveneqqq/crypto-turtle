from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from services import xai_service

router = APIRouter(prefix="/xai", tags=["xai"])


class ExplainModelRequest(BaseModel):
    model_folder: str
    model_file: str
    explainer_type: str
    data_folder: str
    data_file: str
    instance: Optional[int] = None
    feature: Optional[str] = None
    explainer_to_test: Optional[str] = None


@router.post("/explain-model")
def explain_model_route(request: ExplainModelRequest):

    if not all([
        request.model_folder,
        request.model_file,
        request.data_folder,
        request.data_file
    ]):
        raise HTTPException(
            status_code=400,
            detail="model_folder, model_file, data_folder and data_file are required"
        )

    result = xai_service.request_explain_model(
        model_folder=request.model_folder,
        model_file=request.model_file,
        explainer_type=request.explainer_type,
        data_folder=request.data_folder,   
        data_file=request.data_file,       
        instance=request.instance,
        feature=request.feature,
        explainer_to_test=request.explainer_to_test
    )

    return {
        "msg": "Request added to XAI queue",
        "rabbit_msg_id": result["rabbit_msg_id"]
    }
