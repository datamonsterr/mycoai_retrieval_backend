from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from .training_models import (
    RollbackRequest,
    TrainingJob,
    TrainingJobType,
    TrainingPreflight,
    TrainingSummary,
    TriggerTrainingRequest,
)
from .training_service import (
    TrainingJobConflictError,
    TrainingJobNotDeployableError,
    TrainingJobNotFoundError,
    training_service,
)

router = APIRouter(prefix="/api/v1/training", tags=["training"])


@router.get("/status")
def get_training_status() -> TrainingSummary:
    return training_service.status()


@router.get("/jobs")
def list_training_jobs() -> list[TrainingJob]:
    return training_service.jobs()


@router.get("/preflight")
def get_training_preflight(
    job_type: Annotated[
        TrainingJobType, Query()
    ] = TrainingJobType.reindex,
) -> TrainingPreflight:
    return training_service.preflight(job_type)


@router.post("/trigger", status_code=status.HTTP_202_ACCEPTED)
def trigger_training(request: TriggerTrainingRequest) -> TrainingJob:
    try:
        return training_service.trigger(request.job_type, request.trigger)
    except TrainingJobConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Training job already running: {error}",
        ) from error


@router.get("/jobs/{job_id}")
def get_training_job(job_id: str) -> TrainingJob:
    try:
        return training_service.get_job(job_id)
    except TrainingJobNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job not found: {job_id}",
        ) from error


@router.post("/jobs/{job_id}/cancel")
def cancel_training_job(job_id: str) -> TrainingJob:
    try:
        return training_service.cancel(job_id)
    except TrainingJobNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job not found: {job_id}",
        ) from error


@router.post("/jobs/{job_id}/deploy")
def deploy_training_job(job_id: str) -> TrainingJob:
    try:
        return training_service.deploy(job_id)
    except TrainingJobNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job not found: {job_id}",
        ) from error
    except TrainingJobNotDeployableError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Training job not deployable: {job_id}",
        ) from error


@router.post("/rollback", status_code=status.HTTP_202_ACCEPTED)
def rollback_training(request: RollbackRequest) -> TrainingJob:
    try:
        return training_service.rollback(request)
    except TrainingJobNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version not found: {request.version}",
        ) from error
