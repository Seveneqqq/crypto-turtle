from fastapi import FastAPI

app = FastAPI() 

@app.get("/")
def hello_func():
    return {"Hello": "World"}