from fastapi import APIRouter

from mycoai_retrieval_backend.schemas.training import TrainingJobRead, TrainingTrigger

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/trigger")
async def trigger_training(payload: TrainingTrigger) -> TrainingJobRead:
    return TrainingJobRead(job_id="pending", status="queued", reason=payload.reason)


@router.get("/status/{job_id}")
async def training_status(job_id: str) -> TrainingJobRead:
    return TrainingJobRead(job_id=job_id, status="queued", reason=None)
