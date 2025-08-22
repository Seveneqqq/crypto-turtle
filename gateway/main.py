from fastapi import FastAPI
from routes.auth_routes import router as auth_router
from routes.data_routes import router as data_router
from starlette.middleware.sessions import SessionMiddleware
from configuration.configuration import SESSION_MIDDLEWARE_SECRET
    
app = FastAPI()
app.include_router(auth_router)
app.include_router(data_router)
app.add_middleware(SessionMiddleware, secret_key=SESSION_MIDDLEWARE_SECRET)

@app.get("/")
def root():
    return {"message": "Hello World"}

