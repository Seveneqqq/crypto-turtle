from fastapi import APIRouter, File, Request, UploadFile
from fastapi.params import Depends
from services import auth_service
from services import file_service

router = APIRouter(prefix="/file", tags=["file"])

@router.get("/private")
def private_data(
    user = Depends(auth_service.require_auth)
    ):
    return {"msg": f"Witaj w data, {user.user_metadata.get('name')}"}


@router.post("")
def upload_file(
    user = Depends(auth_service.require_auth), 
    file: UploadFile = File(...)
):
    return file_service.upload_file(file)

@router.get("")
def get_files(
    user = Depends(auth_service.require_auth)
):
    return file_service.get_all_files()

@router.delete("/{filename}")
def delete_file(
    user = Depends(auth_service.require_auth),
    filename: str = None
):
    return file_service.delete_file(filename)
