from fastapi import FastAPI
from routes.auth_routes import router as auth_router
from starlette.middleware.sessions import SessionMiddleware
from configuration.configuration import SESSION_MIDDLEWARE_SECRET
    
app = FastAPI()
app.include_router(auth_router)
app.add_middleware(SessionMiddleware, secret_key=SESSION_MIDDLEWARE_SECRET)

@app.get("/")
def root():
    return {"message": "Hello World"}

