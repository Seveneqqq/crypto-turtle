from fastapi import APIRouter, File, Request, UploadFile
from fastapi.params import Depends
from services import auth_service
from services import data_service

router = APIRouter(prefix="/data", tags=["data"])

@router.get("/private")
def private_data(
    user = Depends(auth_service.require_auth)
    ):
    return {"msg": f"Witaj w data, {user.user_metadata.get('name')}"}


@router.post("/file")
def upload_file(
    user = Depends(auth_service.require_auth), 
    file: UploadFile = File(...)
):
    return data_service.upload_file(file)

@router.get("/file")
def get_files(
    user = Depends(auth_service.require_auth)
):
    return data_service.get_all_files()

@router.delete("/file/{filename}")
def delete_file(
    user = Depends(auth_service.require_auth),
    filename: str = None
):
    return data_service.delete_file(filename)
