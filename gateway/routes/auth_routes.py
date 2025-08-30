from fastapi import APIRouter, Request, Depends
from services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])

@router.get("/login")
def login(request: Request):
    next_url = request.query_params.get("next") or request.headers.get("referer") or "/"
    return auth_service.login_with_google(next_url)

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