from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
import httpx
import os
from supabase import create_client, Client

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

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


@app.get("/")
def hello_func():
    return {"Hello": "World"}

@app.get("/login")
def login():
    # Redirect użytkownika do Google OAuth
    redirect_uri = (
        f"{GOOGLE_AUTH_URI}?response_type=code&client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}&scope=email%20profile&access_type=offline&prompt=consent"
    )
    return RedirectResponse(url=redirect_uri)

@app.get("/login/callback")
def login_callback(code: str):
    try:
        # Zamiana kodu na token
        data = {
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code"
        }
        r = httpx.post(GOOGLE_TOKEN_URI, data=data)
        token_data = r.json()
        access_token = token_data.get("access_token")
        
        if not access_token:
            return {"error": "Nie udało się pobrać tokenu z Google", "details": token_data}

        # Pobranie info o użytkowniku z Google
        headers = {"Authorization": f"Bearer {access_token}"}
        r = httpx.get(GOOGLE_USERINFO_URI, headers=headers)
        user_info = r.json()
        email = user_info.get("email")
        name = user_info.get("name")
        
        if not email:
            return {"error": "Nie udało się pobrać emaila użytkownika"}

        print(f"Próba utworzenia użytkownika: {email}")
        
        # Próba utworzenia użytkownika - jeśli już istnieje, Supabase zwróci błąd
        try:
            create_response = supabase.auth.admin.create_user({
                "email": email,
                "password": os.urandom(16).hex(),  # losowe hasło
                "email_confirm": True,
                "user_metadata": {"name": name}
            })
            
            print(f"Create response: {create_response}")
            
            if create_response.user:
                return {
                    "email": email, 
                    "name": name, 
                    "message": "Nowy użytkownik utworzony i zalogowany",
                    "user_id": create_response.user.id
                }
            else:
                return {"error": "Nie udało się utworzyć użytkownika", "details": str(create_response)}
                
        except Exception as create_error:
            print(f"Create error: {create_error}")
            error_str = str(create_error).lower()
            
            # Sprawdź czy to błąd "użytkownik już istnieje"
            if any(phrase in error_str for phrase in ["already registered", "already exists", "duplicate", "user_email_exists"]):
                return {
                    "email": email, 
                    "name": name, 
                    "message": "Użytkownik już istnieje - zalogowany pomyślnie"
                }
            else:
                # Inny błąd
                return {"error": "Błąd podczas tworzenia użytkownika", "details": str(create_error)}
            
    except Exception as e:
        return {"error": "Wystąpił błąd podczas logowania", "details": str(e)}