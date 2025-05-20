# backend.py
from fastapi import FastAPI
from frontend import app as nicegui_app  # Import NiceGUI's FastAPI app

app = FastAPI()

# Mount NiceGUI at root or subpath
app.mount("/", nicegui_app)
# Optional: app.mount("/deenai", nicegui_app)  # Then routes become /deenai/chat etc.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
