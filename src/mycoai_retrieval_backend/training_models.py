from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class TrainingJobType(StrEnum):
    reindex = "reindex"
    finetune = "finetune"
    full_retrain = "full_retrain"


class TrainingStatus(StrEnum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TrainingStage(StrEnum):
    preparing = "preparing"
    extracting = "extracting"
    training = "training"
    evaluating = "evaluating"
    indexing = "indexing"


class TrainingTrigger(StrEnum):
    manual = "manual"
    scheduled = "scheduled"
    feedback_accepted = "feedback_accepted"


class TrainingChanges(BaseModel):
    strains_added: int = Field(ge=0)
    strains_archived: int = Field(ge=0)
    feedback_accepted: int = Field(ge=0)


class TrainingProgress(BaseModel):
    stage: TrainingStage
    current: int = Field(ge=0)
    total: int = Field(gt=0)
    epoch: int | None = Field(default=None, ge=0)
    current_loss: float | None = Field(default=None, ge=0)
    current_accuracy: float | None = Field(default=None, ge=0, le=1)
    estimated_seconds_remaining: int | None = Field(default=None, ge=0)


class TrainingMetrics(BaseModel):
    f1_score: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)


class ModelInfo(BaseModel):
    version: str
    name: str
    last_training_date: datetime
    strains_in_training_set: int = Field(ge=0)
    metrics: TrainingMetrics
    staged_version: str | None = None
    previous_versions: list[str]


class TrainingJob(BaseModel):
    job_id: str
    job_type: TrainingJobType
    status: TrainingStatus
    progress: TrainingProgress
    trigger: TrainingTrigger
    changes_since_last: TrainingChanges
    started_at: datetime
    completed_at: datetime | None = None
    estimated_completion: datetime | None = None
    model_version: str | None = None
    staged_model_version: str | None = None
    metrics: TrainingMetrics | None = None
    log: list[str]
    notification: str | None = None
    deployment_status: Literal["not_deployable", "staged", "deployed", "rolled_back"]


class TrainingSummary(BaseModel):
    current_model: ModelInfo
    active_job: TrainingJob | None
    latest_job: TrainingJob | None
    pending_notification: str | None


class TrainingPreflight(BaseModel):
    job_type: TrainingJobType
    changes_since_last: TrainingChanges
    estimated_training_hours: float
    can_start: bool
    blocking_job_id: str | None = None


class TriggerTrainingRequest(BaseModel):
    job_type: TrainingJobType = TrainingJobType.reindex
    trigger: TrainingTrigger = TrainingTrigger.manual


class RollbackRequest(BaseModel):
    version: str
