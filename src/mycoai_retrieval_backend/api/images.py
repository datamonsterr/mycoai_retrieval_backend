from fastapi import APIRouter, UploadFile

from mycoai_retrieval_backend.schemas.images import (
    BatchUploadResponse,
    ImageUploadResponse,
)

router = APIRouter(prefix="/images", tags=["images"])


@router.post("/upload")
async def upload_image(file: UploadFile) -> ImageUploadResponse:
    return ImageUploadResponse(filename=file.filename or "upload", status="accepted")


@router.post("/batch")
async def batch_upload(files: list[UploadFile]) -> BatchUploadResponse:
    return BatchUploadResponse(job_id="pending", accepted=len(files))
