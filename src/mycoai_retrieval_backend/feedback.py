from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class FeedbackSource(StrEnum):
    query_result = "query_result"
    database_review = "database_review"


class FeedbackStatus(StrEnum):
    pending = "pending"
    accepted = "accepted"
    rejected = "rejected"
    deferred = "deferred"


class NotificationType(StrEnum):
    feedback_submitted = "feedback_submitted"
    feedback_accepted = "feedback_accepted"
    feedback_rejected = "feedback_rejected"


class FeedbackCreate(BaseModel):
    submitter_id: UUID
    source: FeedbackSource
    suggested_species: str = Field(min_length=1)
    description: str = Field(min_length=1)
    query_strain: str | None = None
    result_id: UUID | None = None
    image_id: UUID | None = None
    predicted_species: str | None = None
    supporting_image_url: str | None = None

    @model_validator(mode="after")
    def validate_source_target(self) -> "FeedbackCreate":
        if self.source is FeedbackSource.query_result and not self.query_strain:
            raise ValueError("query_strain is required for query_result feedback")
        if self.source is FeedbackSource.database_review and self.image_id is None:
            raise ValueError("image_id is required for database_review feedback")
        return self


class FeedbackReview(BaseModel):
    reviewer_id: UUID
    status: FeedbackStatus
    review_note: str | None = None

    @model_validator(mode="after")
    def validate_review_note(self) -> "FeedbackReview":
        if self.status is FeedbackStatus.rejected and not self.review_note:
            raise ValueError("review_note is required when rejecting feedback")
        return self


class FeedbackBatchReview(FeedbackReview):
    feedback_ids: list[UUID] = Field(min_length=1)


class Notification(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    type: NotificationType
    feedback_id: UUID
    message: str
    read: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AuditLogEntry(BaseModel):
    id: int
    user_id: UUID
    action: str
    entity_type: str
    entity_id: UUID
    changes: dict[str, object]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReindexTask(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    feedback_id: UUID
    strain: str
    status: str = "queued"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FeedbackRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    submitter_id: UUID
    reviewer_id: UUID | None = None
    source: FeedbackSource
    query_strain: str | None = None
    result_id: UUID | None = None
    image_id: UUID | None = None
    predicted_species: str | None = None
    suggested_species: str
    description: str
    status: FeedbackStatus = FeedbackStatus.pending
    review_note: str | None = None
    supporting_image_url: str | None = None
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reviewed_at: datetime | None = None
    reindex_task_id: UUID | None = None


class FeedbackStats(BaseModel):
    total_by_source: dict[FeedbackSource, int]
    acceptance_rate: float
    average_review_seconds: float | None
    most_misclassified_species: list[tuple[str, int]]
    feedback_by_strain: list[tuple[str, int]]
