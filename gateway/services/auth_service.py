from http.client import HTTPException
from typing import Optional
from httpcore import request
from supabase import create_client, Client
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi import HTTPException, status, Depends, Request
from starlette.middleware.sessions import SessionMiddleware

class AuthService:
    
    def __init__(self, SUPABASE_URL: str, SUPABASE_KEY: str, GOOGLE_REDIRECT_URI: str):
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.GOOGLE_REDIRECT_URI = GOOGLE_REDIRECT_URI

    def login_with_google(self):
        response = self.supabase.auth.sign_in_with_oauth({
            "provider": "google",
            "options": {
                "redirect_to": self.GOOGLE_REDIRECT_URI,
                "scopes": "email profile",
                "flow_type": "pkce",
                "query_params": {"prompt": "consent"}
            }
        })
        return RedirectResponse(url=response.url)

    from fastapi.responses import JSONResponse

    def login_callback(self, request: Request):
        code = request.query_params.get("code")
        response = self.supabase.auth.exchange_code_for_session({
            "auth_code": code
        })
    
        access_token = response.session.access_token
        refresh_token = response.session.refresh_token
    
        user_data = {
            "id": response.session.user.id,
            "email": response.session.user.email,
            "name": response.session.user.user_metadata.get("name"),
        }
    
        res = JSONResponse(
            content={
                "user": user_data
            }
        )
    
        res.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            secure=False,   
            samesite="lax",
            max_age=3600
        )
        res.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=True,
            secure=False,
            samesite="lax",
            max_age=60*60*24*14
        )
    
        return res

    def require_auth(self, request: Request):
        try:
            return self.get_current_user(request)
        except HTTPException:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Nieautoryzowany")

    def require_auth(self, request: Request):
        user = self.get_current_user(request)
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Nieautoryzowany")
        return user

    def get_current_user(self, request: Request):
        auth_header: Optional[str] = request.headers.get("Authorization")
        token: Optional[str] = None

        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")

        if not token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Brak tokena")

        user = self.supabase.auth.get_user(token)
        if not user.user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Nieautoryzowany")

        return user.user
    
    
    def get_current_user_dep(self, request: Request):
        return self.get_current_user(request)