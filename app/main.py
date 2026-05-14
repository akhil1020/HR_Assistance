import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.api.upload_api import router as upload_router
from app.api.chatbot_api import router as chatbot_router

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Pre-warm heavy resources once after server starts — not at import time
    print("Loading embedding model…")
    from app.data_pipeline import get_embeddings
    get_embeddings()
    print("Connecting to Pinecone…")
    from app.api.chatbot_api import _get_vector_store, llm
    _get_vector_store()
    print("Warming up Ollama model…")
    llm.invoke("hi")   # forces Ollama to load the model into memory now
    print("Ready.")
    yield


app = FastAPI(
    title="HR Assistance API",
    description="API for HR assistance and document processing with Ollama chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api/upload", tags=["upload"])
app.include_router(chatbot_router, prefix="/api/chat", tags=["chatbot"])

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
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["app"],   # only watch app/ — ignores .venv
    )

if __name__ == "__main__":
    main()
