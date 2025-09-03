from fastapi import APIRouter, File, Request, UploadFile
from fastapi.params import Depends
from services import auth_service
from services import sqlite_service


router = APIRouter(prefix="/tasks-history", tags=["tasks-history"])

@router.get("/")
def get_historical_data(
    action: str = None,
):
    response = sqlite_service.get_messages_by_action(
        action=action
    )
    return {"Tasks": response}

