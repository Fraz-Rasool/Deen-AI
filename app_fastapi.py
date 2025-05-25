from fastapi import FastAPI
import uvicorn
from main import app as niceguiapp

app = FastAPI()
app.mount("/",niceguiapp)
if __name__=="__main__":
    uvicorn.run(app, host="127.0.0.1",port=8000)