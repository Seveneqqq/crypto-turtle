from fastapi import APIRouter, Request
from fastapi.params import Depends
from services import auth_service

router = APIRouter(prefix="/data", tags=["data"])

@router.get("/private")
def private_data(user = Depends(auth_service.require_auth)):
    return {"msg": f"Witaj w data, {user.user_metadata.get('name')}"}