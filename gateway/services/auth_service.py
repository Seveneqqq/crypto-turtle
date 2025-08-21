from supabase import create_client, Client
from fastapi.responses import RedirectResponse
import httpx
import os

class Auth_Service:

    def __init__(self, GOOGLE_AUTH_URI, GOOGLE_REDIRECT_URI, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_USERINFO_URI, GOOGLE_TOKEN_URI, SUPABASE_URL, SUPABASE_KEY):
        self.GOOGLE_AUTH_URI = GOOGLE_AUTH_URI
        self.GOOGLE_REDIRECT_URI = GOOGLE_REDIRECT_URI
        self.GOOGLE_CLIENT_ID = GOOGLE_CLIENT_ID
        self.GOOGLE_CLIENT_SECRET = GOOGLE_CLIENT_SECRET
        self.GOOGLE_USERINFO_URI = GOOGLE_USERINFO_URI
        self.GOOGLE_TOKEN_URI = GOOGLE_TOKEN_URI
        self.SUPABASE_URL = SUPABASE_URL
        self.SUPABASE_KEY = SUPABASE_KEY
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def login(self):
    # Redirect użytkownika do Google OAuth
        redirect_uri = (
            f"{self.GOOGLE_AUTH_URI}?response_type=code&client_id={self.GOOGLE_CLIENT_ID}"
            f"&redirect_uri={self.GOOGLE_REDIRECT_URI}&scope=email%20profile&access_type=offline&prompt=consent"
        )
        return RedirectResponse(url=redirect_uri)
    
    def login_callback(self, code):
        try:
            # Zamiana kodu na token
            data = {
                "code": code,
                "client_id": self.GOOGLE_CLIENT_ID,
                "client_secret": self.GOOGLE_CLIENT_SECRET,
                "redirect_uri": self.GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code"
            }
            r = httpx.post(self.GOOGLE_TOKEN_URI, data=data)
            token_data = r.json()
            access_token = token_data.get("access_token")

            if not access_token:
                return {"error": "Nie udało się pobrać tokenu z Google", "details": token_data}

            # Pobranie info o użytkowniku z Google
            headers = {"Authorization": f"Bearer {access_token}"}
            r = httpx.get(self.GOOGLE_USERINFO_URI, headers=headers)
            user_info = r.json()
            email = user_info.get("email")
            name = user_info.get("name")

            if not email:
                return {"error": "Nie udało się pobrać emaila użytkownika"}

            print(f"Próba utworzenia użytkownika: {email}")

            # Próba utworzenia użytkownika - jeśli już istnieje, Supabase zwróci błąd
            try:
                create_response = self.supabase.auth.admin.create_user({
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