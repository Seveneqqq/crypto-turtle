from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
import httpx
import os
from supabase import create_client, Client
from services.auth_service import Auth_Service

from configuration.configuration import (
    SUPABASE_URL,
    SUPABASE_KEY,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REDIRECT_URI,
    GOOGLE_AUTH_URI,
    GOOGLE_TOKEN_URI,
    GOOGLE_USERINFO_URI
)


app = FastAPI()

auth_service = Auth_Service(
    GOOGLE_AUTH_URI,
    GOOGLE_REDIRECT_URI,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_USERINFO_URI,
    GOOGLE_TOKEN_URI,
    SUPABASE_URL,
    SUPABASE_KEY
)


@app.get("/")
def hello_func():
    return {"Hello": "World"}

@app.get("/login")
def login(): 
    return auth_service.login()

@app.get("/login/callback")
def login_callback(request: Request):
    code = request.query_params.get("code")
    return auth_service.login_callback(code) 