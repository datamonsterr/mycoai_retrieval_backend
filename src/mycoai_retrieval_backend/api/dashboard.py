from fastapi import APIRouter

from mycoai_retrieval_backend.schemas.dashboard import DashboardStats

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats")
async def stats() -> DashboardStats:
    return DashboardStats(images=0, species=0, feedback_items=0, training_jobs=0)
