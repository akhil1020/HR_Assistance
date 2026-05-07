import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.api.upload_api import router as upload_router
from app.api.chatbot_api import router as chatbot_router

app = FastAPI(
    title="HR Assistance API",
    description="API for HR assistance and document processing with Ollama chatbot",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(upload_router, prefix="/api/upload", tags=["upload"])
app.include_router(chatbot_router, prefix="/api/chat", tags=["chatbot"])

# Static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def chat_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/upload")
def upload_ui():
    return FileResponse(str(STATIC_DIR / "upload.html"))

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
