from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


class FeedbackCreate(BaseModel):
    source: str = Field(pattern=r"^(query_result|database_review)$")
    query_strain: str = Field(min_length=1)
    predicted_species: str = Field(min_length=1)
    suggested_species: str = Field(min_length=1)
    description: str = Field(min_length=1)
    submitter_id: str = Field(min_length=1)


class FeedbackItem(BaseModel):
    feedback_id: str
    source: str
    query_strain: str
    predicted_species: str
    suggested_species: str
    description: str
    submitter_id: str
    status: str
    created_at: str
    reviewed_at: str | None = None
    reviewed_by: str | None = None
    review_note: str | None = None


class FeedbackReview(BaseModel):
    action: str = Field(pattern=r"^(accept|reject)$")
    reviewed_by: str = Field(min_length=1)
    review_note: str | None = None
    feedback_ids: list[str] = Field(default_factory=list)


_store: dict[str, FeedbackItem] = {}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


@router.post("/", response_model=FeedbackItem, status_code=201)
def submit_feedback(body: FeedbackCreate) -> FeedbackItem:
    item = FeedbackItem(
        feedback_id=str(uuid.uuid4()),
        source=body.source,
        query_strain=body.query_strain,
        predicted_species=body.predicted_species,
        suggested_species=body.suggested_species,
        description=body.description,
        submitter_id=body.submitter_id,
        status="pending",
        created_at=_now_iso(),
    )
    _store[item.feedback_id] = item
    return item


@router.get("/", response_model=list[FeedbackItem])
def list_feedback(
    status: str | None = None,
    submitter_id: str | None = None,
    source: str | None = None,
) -> list[FeedbackItem]:
    results: list[FeedbackItem] = list(_store.values())
    if status is not None:
        results = [r for r in results if r.status == status]
    if submitter_id is not None:
        results = [r for r in results if r.submitter_id == submitter_id]
    if source is not None:
        results = [r for r in results if r.source == source]
    results.sort(key=lambda r: r.created_at, reverse=True)
    return results


@router.patch("/review/bulk", response_model=list[FeedbackItem])
def review_feedback_bulk(body: FeedbackReview) -> list[FeedbackItem]:
    reviewed: list[FeedbackItem] = []
    for fid in body.feedback_ids:
        item = _store.get(fid)
        if item is None or item.status != "pending":
            continue
        item.status = "accepted" if body.action == "accept" else "rejected"
        item.reviewed_at = _now_iso()
        item.reviewed_by = body.reviewed_by
        item.review_note = body.review_note
        reviewed.append(item)
    return reviewed


@router.get("/{feedback_id}", response_model=FeedbackItem)
def get_feedback(feedback_id: str) -> FeedbackItem:
    item = _store.get(feedback_id)
    if item is None:
        raise HTTPException(status_code=404, detail="feedback not found")
    return item


@router.patch("/{feedback_id}/review", response_model=FeedbackItem)
def review_feedback(feedback_id: str, body: FeedbackReview) -> FeedbackItem:
    item = _store.get(feedback_id)
    if item is None:
        raise HTTPException(status_code=404, detail="feedback not found")
    if item.status != "pending":
        raise HTTPException(status_code=409, detail=f"feedback already {item.status}")
    item.status = "accepted" if body.action == "accept" else "rejected"
    item.reviewed_at = _now_iso()
    item.reviewed_by = body.reviewed_by
    item.review_note = body.review_note
    return item
