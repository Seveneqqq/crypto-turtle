from fastapi import APIRouter, Request
from fastapi.params import Depends
from services.auth_service import AuthService
from configuration.configuration import SUPABASE_URL, SUPABASE_KEY, GOOGLE_REDIRECT_URI

auth_service = AuthService(SUPABASE_URL, SUPABASE_KEY, GOOGLE_REDIRECT_URI)
router = APIRouter(prefix="/auth", tags=["auth"])

@router.get("/login")
def login():
    return auth_service.login_with_google()

@router.get("/login/callback")
def callback(request: Request):
    return auth_service.login_callback(request)

@router.get("/me")
def read_me(current_user = Depends(auth_service.get_current_user_dep)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.user_metadata.get("name")
    }

@router.get("/private")
def private_data(user = Depends(auth_service.require_auth)):
    return {"msg": f"Witaj, {user.user_metadata.get('name')}"}