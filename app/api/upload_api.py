from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List
from fastapi.responses import Response
import os
from data_pipeline import process_uploaded_file

# Create router for upload endpoints
router = APIRouter()


UPLOAD_DIR = "app/data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    # Extract extension
    _, ext = os.path.splitext(file.filename.lower())

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

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Trigger pipeline in background
    background_tasks.add_task(process_uploaded_file, file_path)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "status": "uploaded",
        "message": "File uploaded successfully. Vector database is being updated in background."
    }

