from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List
from fastapi.responses import Response
from pathlib import Path
from app.data_pipeline import process_uploaded_file

# Create router for upload endpoints
router = APIRouter()

# Configuration - Using pathlib for cross-platform path support
BASE_DIR = Path(__file__).parent.parent.parent  # Project root
UPLOAD_DIR = BASE_DIR / "app" / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    # Extract extension using pathlib
    file_path_obj = Path(file.filename)
    ext = file_path_obj.suffix.lower()

    # Validate extension
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are allowed"
        )

    # Validate MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type"
        )

    file_path = UPLOAD_DIR / file.filename

    # Save file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Trigger pipeline in background
    background_tasks.add_task(process_uploaded_file, str(file_path))

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "status": "uploaded",
        "message": "File uploaded successfully. Vector database is being updated in background."
    }

