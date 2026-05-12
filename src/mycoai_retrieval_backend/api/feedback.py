from fastapi import APIRouter

from mycoai_retrieval_backend.schemas.feedback import FeedbackCreate, FeedbackItem

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("")
async def create_feedback(payload: FeedbackCreate) -> FeedbackItem:
    return FeedbackItem(id="pending", image_id=payload.image_id, label=payload.label)


@router.get("/inbox")
async def inbox() -> list[FeedbackItem]:
    return []
