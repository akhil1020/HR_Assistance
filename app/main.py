import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.upload_api import router as upload_router

# Initialize FastAPI app
app = FastAPI(
    title="HR Assistance API",
    description="API for HR assistance and document processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload_router, prefix="/api", tags=["upload"])

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "HR Assistance API", "version": "1.0.0"}

def main():
    """Run the FastAPI server"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main()
