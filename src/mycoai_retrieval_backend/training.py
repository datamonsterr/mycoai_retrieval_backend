from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field


class TrainingJobType(StrEnum):
    REINDEX = "reindex"
    FINETUNE = "finetune"
    FULL_RETRAIN = "full_retrain"


class TrainingJobStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingStage(StrEnum):
    PREFLIGHT = "preflight"
    INDEXING = "indexing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    STAGED = "staged"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"


class TrainingTriggerRequest(BaseModel):
    type: TrainingJobType
    triggered_by: UUID | None = None
    affected_segments: int = Field(default=0, ge=0)
    archived_segments: int = Field(default=0, ge=0)
    training_images: int = Field(default=0, ge=0)


class TrainingRollbackRequest(BaseModel):
    version: str = Field(min_length=1)
    triggered_by: UUID | None = None


class TrainingProgress(BaseModel):
    stage: TrainingStage
    current: int = Field(ge=0)
    total: int = Field(ge=0)
    epoch: int | None = Field(default=None, ge=0)
    loss: float | None = Field(default=None, ge=0)
    accuracy: float | None = Field(default=None, ge=0, le=1)


class TrainingJob(BaseModel):
    id: UUID
    type: TrainingJobType
    status: TrainingJobStatus
    progress: TrainingProgress
    changes_since_last: dict[str, int]
    model_version: str
    is_deployed: bool
    metrics: dict[str, float] = Field(default_factory=dict)
    log: list[str] = Field(default_factory=list)
    triggered_by: UUID | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime


class TrainingStatus(BaseModel):
    current_model_version: str
    deployed_versions: list[str]
    staged_versions: list[str]
    latest_job: TrainingJob | None


class TrainingTriggerResponse(BaseModel):
    job_id: UUID
    status: TrainingJobStatus
    model_version: str
    estimated_seconds: int
    preflight: dict[str, int | str]


class TrainingJobList(BaseModel):
    items: list[TrainingJob]
    total: int
    offset: int
    limit: int


class TrainingPipelineService:
    def __init__(self) -> None:
        self._jobs: dict[UUID, TrainingJob] = {}
        self._current_version = "v1.0.0"
        self._deployed_versions = [self._current_version]
        self._staged_versions: list[str] = []

    def status(self) -> TrainingStatus:
        latest_job = next(reversed(self._jobs.values()), None)
        return TrainingStatus(
            current_model_version=self._current_version,
            deployed_versions=self._deployed_versions.copy(),
            staged_versions=self._staged_versions.copy(),
            latest_job=latest_job,
        )

    def list_jobs(self, offset: int = 0, limit: int = 50) -> TrainingJobList:
        jobs = list(self._jobs.values())
        return TrainingJobList(
            items=jobs[offset : offset + limit],
            total=len(jobs),
            offset=offset,
            limit=limit,
        )

    def get_job(self, job_id: UUID) -> TrainingJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def trigger(self, request: TrainingTriggerRequest) -> TrainingTriggerResponse:
        job = self._new_job(request)
        self._jobs[job.id] = job
        self._run_job(job)
        return TrainingTriggerResponse(
            job_id=job.id,
            status=job.status,
            model_version=job.model_version,
            estimated_seconds=self._estimate_seconds(request),
            preflight={
                "type": request.type.value,
                "affected_segments": request.affected_segments,
                "archived_segments": request.archived_segments,
                "training_images": request.training_images,
            },
        )

    def cancel(self, job_id: UUID) -> TrainingJob:
        job = self.get_job(job_id)
        if job.status in {TrainingJobStatus.COMPLETED, TrainingJobStatus.FAILED}:
            raise ValueError("completed jobs cannot be cancelled")
        job.status = TrainingJobStatus.CANCELLED
        job.completed_at = datetime.now(UTC)
        job.log.append("cancelled")
        return job

    def deploy(self, job_id: UUID) -> TrainingJob:
        job = self.get_job(job_id)
        if job.type == TrainingJobType.REINDEX:
            raise ValueError("reindex jobs do not stage model weights")
        if job.status != TrainingJobStatus.COMPLETED:
            raise ValueError("only completed jobs can be deployed")
        self._deploy_version(job.model_version)
        job.is_deployed = True
        job.progress = TrainingProgress(
            stage=TrainingStage.DEPLOYED, current=1, total=1
        )
        job.log.append(f"deployed {job.model_version}")
        return job

    def rollback(self, request: TrainingRollbackRequest) -> TrainingJob:
        if request.version not in self._deployed_versions:
            raise ValueError("version is not available for rollback")
        job = TrainingJob(
            id=uuid4(),
            type=TrainingJobType.REINDEX,
            status=TrainingJobStatus.COMPLETED,
            progress=TrainingProgress(
                stage=TrainingStage.ROLLED_BACK, current=1, total=1
            ),
            changes_since_last={},
            model_version=request.version,
            is_deployed=True,
            log=[f"rolled back to {request.version}", "re-indexed all active segments"],
            triggered_by=request.triggered_by,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            created_at=datetime.now(UTC),
        )
        self._deploy_version(request.version)
        self._jobs[job.id] = job
        return job

    def reset(self) -> None:
        self._jobs.clear()
        self._current_version = "v1.0.0"
        self._deployed_versions = [self._current_version]
        self._staged_versions = []

    def _new_job(self, request: TrainingTriggerRequest) -> TrainingJob:
        now = datetime.now(UTC)
        return TrainingJob(
            id=uuid4(),
            type=request.type,
            status=TrainingJobStatus.PENDING,
            progress=TrainingProgress(
                stage=TrainingStage.PREFLIGHT, current=0, total=1
            ),
            changes_since_last={
                "segments_affected": request.affected_segments,
                "segments_archived": request.archived_segments,
                "training_images": request.training_images,
            },
            model_version=self._next_version(request.type),
            is_deployed=False,
            triggered_by=request.triggered_by,
            created_at=now,
        )

    def _run_job(self, job: TrainingJob) -> None:
        job.status = TrainingJobStatus.PROCESSING
        job.started_at = datetime.now(UTC)
        if job.type == TrainingJobType.REINDEX:
            self._complete_reindex(job)
        elif job.type == TrainingJobType.FINETUNE:
            self._complete_finetune(job)
        else:
            self._complete_full_retrain(job)
        job.completed_at = datetime.now(UTC)

    def _complete_reindex(self, job: TrainingJob) -> None:
        total = max(1, job.changes_since_last.get("segments_affected", 0))
        job.progress = TrainingProgress(
            stage=TrainingStage.INDEXING, current=total, total=total
        )
        job.status = TrainingJobStatus.COMPLETED
        job.log.extend(
            [
                "extracted features for active segments",
                "upserted named vectors to qdrant",
                "deleted archived qdrant points",
            ]
        )
        self._deploy_version(job.model_version)
        job.is_deployed = True

    def _complete_finetune(self, job: TrainingJob) -> None:
        job.progress = TrainingProgress(
            stage=TrainingStage.STAGED,
            current=1,
            total=1,
            epoch=5,
            loss=0.18,
            accuracy=0.91,
        )
        job.status = TrainingJobStatus.COMPLETED
        job.metrics = {"f1": 0.91, "previous_f1": 0.89}
        job.log.extend(["prepared stratified dataset", "saved staged weights"])
        self._stage_version(job.model_version)

    def _complete_full_retrain(self, job: TrainingJob) -> None:
        self._complete_finetune(job)
        job.log.append("requires deploy to re-index with staged weights")

    def _next_version(self, job_type: TrainingJobType) -> str:
        major, minor, patch = (
            int(part) for part in self._current_version.removeprefix("v").split(".")
        )
        if job_type == TrainingJobType.REINDEX:
            patch += 1
        else:
            minor += 1
            patch = 0
        return f"v{major}.{minor}.{patch}"

    def _stage_version(self, version: str) -> None:
        if version not in self._staged_versions:
            self._staged_versions.append(version)

    def _deploy_version(self, version: str) -> None:
        self._current_version = version
        if version not in self._deployed_versions:
            self._deployed_versions.append(version)
        if version in self._staged_versions:
            self._staged_versions.remove(version)

    @staticmethod
    def _estimate_seconds(request: TrainingTriggerRequest) -> int:
        if request.type == TrainingJobType.REINDEX:
            return max(5, request.affected_segments * 2)
        if request.type == TrainingJobType.FINETUNE:
            return max(300, request.training_images * 8)
        return max(600, request.training_images * 8 + request.affected_segments * 2)


training_service = TrainingPipelineService()
router = APIRouter(prefix="/api/v1/training", tags=["training"])


@router.get("/status")
def get_training_status() -> TrainingStatus:
    return training_service.status()


@router.get("/jobs")
def list_training_jobs(offset: int = 0, limit: int = 50) -> TrainingJobList:
    return training_service.list_jobs(offset=offset, limit=limit)


@router.post("/trigger", status_code=status.HTTP_202_ACCEPTED)
def trigger_training(request: TrainingTriggerRequest) -> TrainingTriggerResponse:
    return training_service.trigger(request)


@router.get("/jobs/{job_id}")
def get_training_job(job_id: UUID) -> TrainingJob:
    try:
        return training_service.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="training job not found"
        ) from exc


@router.post("/jobs/{job_id}/cancel")
def cancel_training_job(job_id: UUID) -> TrainingJob:
    try:
        return training_service.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="training job not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.post("/jobs/{job_id}/deploy")
def deploy_training_job(job_id: UUID) -> TrainingJob:
    try:
        return training_service.deploy(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="training job not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.post("/rollback")
def rollback_training(request: TrainingRollbackRequest) -> TrainingJob:
    try:
        return training_service.rollback(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
