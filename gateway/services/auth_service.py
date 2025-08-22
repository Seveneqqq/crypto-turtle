from http.client import HTTPException
from httpcore import request
from supabase import create_client, Client
from fastapi.responses import RedirectResponse
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

    def login_callback(self, request: Request):
        code = request.query_params.get("code")
        response = self.supabase.auth.exchange_code_for_session({
            "auth_code": code
        })
        return response

    def require_auth(self, request: Request):
        try:
            return self.get_current_user(request)
        except HTTPException:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Nieautoryzowany")

    def get_current_user(self, request: Request):
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Brak tokena")

        try:
            token_type, access_token = auth_header.split()
            if token_type.lower() != "bearer":
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Niepoprawny typ tokena")
        except ValueError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Niepoprawny nagłówek Authorization")

        user = self.supabase.auth.get_user(access_token)
        if not user.user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Nieautoryzowany")
        return user.user
    
    
    def get_current_user_dep(self, request: Request):
        return self.get_current_user(request)