from datetime import UTC, datetime, timedelta
from uuid import uuid4

from .training_models import (
    ModelInfo,
    RollbackRequest,
    TrainingChanges,
    TrainingJob,
    TrainingJobType,
    TrainingMetrics,
    TrainingPreflight,
    TrainingProgress,
    TrainingStage,
    TrainingStatus,
    TrainingSummary,
    TrainingTrigger,
)


class TrainingJobConflictError(RuntimeError):
    pass


class TrainingJobNotFoundError(RuntimeError):
    pass


class TrainingJobNotDeployableError(RuntimeError):
    pass


class TrainingService:
    def __init__(self) -> None:
        self._current_model = ModelInfo(
            version="v3.2.1",
            name="EfficientNetB1 finetuned",
            last_training_date=datetime(2026, 5, 1, 10, 30, tzinfo=UTC),
            strains_in_training_set=184,
            metrics=TrainingMetrics(f1_score=0.89, precision=0.91, recall=0.87),
            staged_version="v3.3.0",
            previous_versions=["v3.2.0", "v3.1.0"],
        )
        self._jobs: dict[str, TrainingJob] = {}
        self._seed_history()

    def status(self) -> TrainingSummary:
        return TrainingSummary(
            current_model=self._current_model,
            active_job=self.active_job(),
            latest_job=self.jobs()[0] if self.jobs() else None,
            pending_notification="Model v3.3.0 ready for review",
        )

    def jobs(self) -> list[TrainingJob]:
        return sorted(self._jobs.values(), key=lambda job: job.started_at, reverse=True)

    def get_job(self, job_id: str) -> TrainingJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise TrainingJobNotFoundError(job_id)
        return job

    def preflight(self, job_type: TrainingJobType) -> TrainingPreflight:
        active_job = self.active_job()
        return TrainingPreflight(
            job_type=job_type,
            changes_since_last=TrainingChanges(
                strains_added=12,
                strains_archived=3,
                feedback_accepted=5,
            ),
            estimated_training_hours={
                TrainingJobType.reindex: 0.4,
                TrainingJobType.finetune: 3.5,
                TrainingJobType.full_retrain: 4.2,
            }[job_type],
            can_start=active_job is None,
            blocking_job_id=active_job.job_id if active_job else None,
        )

    def trigger(
        self,
        job_type: TrainingJobType,
        trigger: TrainingTrigger,
    ) -> TrainingJob:
        active_job = self.active_job()
        if active_job is not None:
            raise TrainingJobConflictError(active_job.job_id)

        now = datetime.now(UTC)
        preflight_result = self.preflight(job_type)
        is_ml_job = job_type != TrainingJobType.reindex
        eta_seconds = 3600 if is_ml_job else 1200
        job = TrainingJob(
            job_id=str(uuid4()),
            job_type=job_type,
            status=TrainingStatus.running,
            progress=TrainingProgress(
                stage=TrainingStage.preparing,
                current=1,
                total=50,
                epoch=1 if is_ml_job else None,
                current_loss=0.42 if is_ml_job else None,
                current_accuracy=0.71 if is_ml_job else None,
                estimated_seconds_remaining=eta_seconds,
            ),
            trigger=trigger,
            changes_since_last=preflight_result.changes_since_last,
            started_at=now,
            estimated_completion=now + timedelta(
                hours=preflight_result.estimated_training_hours
            ),
            log=[
                "pre-flight checks passed",
                f"{job_type.value} job started",
                "waiting for current stage checkpoint",
            ],
            deployment_status="not_deployable",
        )
        self._jobs[job.job_id] = job
        return job

    def cancel(self, job_id: str) -> TrainingJob:
        job = self.get_job(job_id)
        if job.status in {TrainingStatus.pending, TrainingStatus.running}:
            job.status = TrainingStatus.cancelled
            job.completed_at = datetime.now(UTC)
            job.log.append("cancel requested; stopped at epoch boundary")
            job.notification = "Training cancelled after graceful checkpoint"
        return job

    def deploy(self, job_id: str) -> TrainingJob:
        job = self.get_job(job_id)
        if job.status != TrainingStatus.completed or job.staged_model_version is None:
            raise TrainingJobNotDeployableError(job_id)
        previous_version = self._current_model.version
        self._current_model.previous_versions = [
            previous_version,
            *self._current_model.previous_versions,
        ]
        self._current_model.version = job.staged_model_version
        self._current_model.staged_version = None
        self._current_model.last_training_date = datetime.now(UTC)
        if job.metrics is not None:
            self._current_model.metrics = job.metrics
        job.deployment_status = "deployed"
        job.notification = f"Model {job.staged_model_version} deployed"
        return job

    def rollback(self, request: RollbackRequest) -> TrainingJob:
        if request.version not in self._current_model.previous_versions:
            raise TrainingJobNotFoundError(request.version)
        now = datetime.now(UTC)
        current_version = self._current_model.version
        prior_versions = [
            v for v in self._current_model.previous_versions
            if v != request.version
        ]
        self._current_model.previous_versions = [
            current_version,
            *prior_versions,
        ]
        self._current_model.version = request.version
        job = TrainingJob(
            job_id=str(uuid4()),
            job_type=TrainingJobType.reindex,
            status=TrainingStatus.completed,
            progress=TrainingProgress(
                stage=TrainingStage.indexing,
                current=50,
                total=50,
                estimated_seconds_remaining=0,
            ),
            trigger=TrainingTrigger.manual,
            changes_since_last=TrainingChanges(
                strains_added=0,
                strains_archived=0,
                feedback_accepted=0,
            ),
            started_at=now,
            completed_at=now,
            model_version=request.version,
            metrics=self._current_model.metrics,
            log=[f"rolled back from {current_version} to {request.version}"],
            notification=f"Rolled back to {request.version}",
            deployment_status="rolled_back",
        )
        self._jobs[job.job_id] = job
        return job

    def active_job(self) -> TrainingJob | None:
        for job in self._jobs.values():
            if job.status in {TrainingStatus.pending, TrainingStatus.running}:
                return job
        return None

    def _seed_history(self) -> None:
        completed_at = datetime(2026, 5, 6, 14, 30, tzinfo=UTC)
        started_at = completed_at - timedelta(hours=3, minutes=45)
        job = TrainingJob(
            job_id="train_20260506_001",
            job_type=TrainingJobType.finetune,
            status=TrainingStatus.completed,
            progress=TrainingProgress(
                stage=TrainingStage.evaluating,
                current=50,
                total=50,
                epoch=25,
                current_loss=0.023,
                current_accuracy=0.91,
                estimated_seconds_remaining=0,
            ),
            trigger=TrainingTrigger.manual,
            changes_since_last=TrainingChanges(
                strains_added=18,
                strains_archived=2,
                feedback_accepted=7,
            ),
            started_at=started_at,
            completed_at=completed_at,
            model_version="v3.3.0",
            staged_model_version="v3.3.0",
            metrics=TrainingMetrics(f1_score=0.92, precision=0.93, recall=0.9),
            log=[
                "dataset prepared",
                "epoch 25/25 loss=0.023 accuracy=0.91",
                "evaluation complete; model staged",
            ],
            notification="Training completed; review staged model",
            deployment_status="staged",
        )
        self._jobs[job.job_id] = job


training_service = TrainingService()
