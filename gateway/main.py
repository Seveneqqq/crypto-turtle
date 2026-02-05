from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware

from routes.auth_routes import router as auth_router
from routes.file_routes import router as file_router
from routes.data_routes import router as data_router
from routes.tests_routes import router as tests_router
from routes.job_routes import router as job_router
from routes.ml_routes import router as ml_router
from routes.rl_routes import router as rl_router
from routes.xai_routes import router as xai_router
from routes.tasks_history_routes import router as tasks_history_router
from configuration.configuration import SESSION_MIDDLEWARE_SECRET

app = FastAPI()

# Dodaj CORS
origins = [
    "http://localhost:5173",  # frontend React
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  
    allow_methods=["*"],    
    allow_headers=["*"],     
)

app.add_middleware(SessionMiddleware, secret_key=SESSION_MIDDLEWARE_SECRET)

app.include_router(auth_router)
app.include_router(file_router)
app.include_router(data_router)
app.include_router(ml_router)
app.include_router(rl_router)
app.include_router(tests_router)
app.include_router(xai_router)
app.include_router(job_router)
app.include_router(tasks_history_router)

@app.get("/")
def root():
    return {"message": "Hello from gateway"}
