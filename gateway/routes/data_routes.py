from fastapi import APIRouter, File, Request, UploadFile
from fastapi.params import Depends
from services import auth_service
from services import data_service

router = APIRouter(prefix="/data", tags=["data"])

@router.get("/private")
def private_data(user = Depends(auth_service.require_auth)):
    return {"msg": f"Witaj w data, {user.user_metadata.get('name')}"}

@router.post("/file")
def upload_file(
    user = Depends(auth_service.require_auth), 
    file: UploadFile = File(...)
):
    return data_service.upload_file(file)