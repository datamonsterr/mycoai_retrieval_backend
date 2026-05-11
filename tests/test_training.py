from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from mycoai_retrieval_backend.app import create_app
from mycoai_retrieval_backend.training import (
    TrainingJob,
    TrainingJobStatus,
    TrainingJobType,
    TrainingPipelineService,
    TrainingProgress,
    TrainingRollbackRequest,
    TrainingStage,
    TrainingTriggerRequest,
)
from mycoai_retrieval_backend.training import (
    training_service as default_training_service,
)


def client_factory(service: TrainingPipelineService | None = None) -> TestClient:
    app = create_app()
    if service is not None:
        from mycoai_retrieval_backend import training as mod

        mod.training_service = service
    else:
        # reset default module service
        if not default_training_service._jobs:
            default_training_service.reset()
    return TestClient(app)


@pytest.fixture
def service() -> TrainingPipelineService:
    return TrainingPipelineService()


@pytest.fixture
def client(service: TrainingPipelineService) -> TestClient:
    return client_factory(service)


@pytest.fixture
def default_client() -> TestClient:
    default_training_service.reset()
    return client_factory()


class TestTrainingPipelineService:
    def test_initial_status(self, service: TrainingPipelineService) -> None:
        status = service.status()
        assert status.current_model_version == "v1.0.0"
        assert status.deployed_versions == ["v1.0.0"]
        assert status.staged_versions == []
        assert status.latest_job is None

    def test_trigger_reindex(self, service: TrainingPipelineService) -> None:
        request = TrainingTriggerRequest(
            type=TrainingJobType.REINDEX,
            affected_segments=10,
            archived_segments=2,
        )
        response = service.trigger(request)
        assert response.status == TrainingJobStatus.COMPLETED
        assert response.model_version == "v1.0.1"
        job = service.get_job(response.job_id)
        assert job.status == TrainingJobStatus.COMPLETED
        assert job.is_deployed
        assert service.status().current_model_version == "v1.0.1"

    def test_trigger_finetune(self, service: TrainingPipelineService) -> None:
        request = TrainingTriggerRequest(
            type=TrainingJobType.FINETUNE,
            training_images=80,
        )
        response = service.trigger(request)
        assert response.status == TrainingJobStatus.COMPLETED
        assert response.model_version == "v1.1.0"
        job = service.get_job(response.job_id)
        assert job.status == TrainingJobStatus.COMPLETED
        assert not job.is_deployed
        assert job.metrics["f1"] == 0.91
        assert "v1.1.0" in service.status().staged_versions
        # current version hasn't changed until deploy
        assert service.status().current_model_version == "v1.0.0"

    def test_trigger_full_retrain(self, service: TrainingPipelineService) -> None:
        request = TrainingTriggerRequest(
            type=TrainingJobType.FULL_RETRAIN,
            training_images=120,
            affected_segments=30,
        )
        response = service.trigger(request)
        assert response.model_version == "v1.1.0"
        assert response.status == TrainingJobStatus.COMPLETED

    def test_deploy_staged_model(self, service: TrainingPipelineService) -> None:
        service.trigger(
            TrainingTriggerRequest(type=TrainingJobType.FINETUNE, training_images=80)
        )
        staged_job = next(reversed(list(service._jobs.values())))
        assert not staged_job.is_deployed
        job = service.deploy(staged_job.id)
        assert job.is_deployed
        assert service.status().current_model_version == "v1.1.0"
        assert "v1.1.0" in service.status().deployed_versions

    def test_cannot_deploy_reindex_job(self, service: TrainingPipelineService) -> None:
        response = service.trigger(
            TrainingTriggerRequest(type=TrainingJobType.REINDEX, affected_segments=5)
        )
        job = service.get_job(response.job_id)
        with pytest.raises(ValueError, match="reindex"):
            service.deploy(job.id)

    def test_cannot_deploy_noncompleted_job(
        self, service: TrainingPipelineService
    ) -> None:
        job = TrainingJob(
            id=uuid4(),
            type=TrainingJobType.FINETUNE,
            status=TrainingJobStatus.PENDING,
            progress=TrainingProgress(
                stage=TrainingStage.PREFLIGHT, current=0, total=1
            ),
            changes_since_last={},
            model_version="v1.2.0",
            is_deployed=False,
            created_at=datetime.now(UTC),
        )
        service._jobs[job.id] = job
        with pytest.raises(ValueError, match="completed"):
            service.deploy(job.id)

    def test_list_jobs(self, service: TrainingPipelineService) -> None:
        service.trigger(
            TrainingTriggerRequest(type=TrainingJobType.REINDEX, affected_segments=1)
        )
        service.trigger(
            TrainingTriggerRequest(type=TrainingJobType.FINETUNE, training_images=80)
        )
        jobs = service.list_jobs()
        assert jobs.total == 2
        assert len(jobs.items) == 2

    def test_cancel_job(self, service: TrainingPipelineService) -> None:
        pending = TrainingJob(
            id=uuid4(),
            type=TrainingJobType.FINETUNE,
            status=TrainingJobStatus.PENDING,
            progress=TrainingProgress(
                stage=TrainingStage.PREFLIGHT, current=0, total=1
            ),
            changes_since_last={},
            model_version="v1.3.0",
            is_deployed=False,
            created_at=datetime.now(UTC),
        )
        service._jobs[pending.id] = pending
        cancelled = service.cancel(pending.id)
        assert cancelled.status == TrainingJobStatus.CANCELLED

    def test_cannot_cancel_completed_job(
        self, service: TrainingPipelineService
    ) -> None:
        response = service.trigger(
            TrainingTriggerRequest(type=TrainingJobType.REINDEX, affected_segments=1)
        )
        job = service.get_job(response.job_id)
        assert job.status == TrainingJobStatus.COMPLETED
        with pytest.raises(ValueError, match="completed"):
            service.cancel(job.id)

    def test_rollback(self, service: TrainingPipelineService) -> None:
        # advance to v1.0.1 via reindex (auto-deploys)
        service.trigger(
            TrainingTriggerRequest(type=TrainingJobType.REINDEX, affected_segments=5)
        )
        assert service.status().current_model_version == "v1.0.1"

        # rollback to v1.0.0
        job = service.rollback(TrainingRollbackRequest(version="v1.0.0"))
        assert job.is_deployed
        assert job.model_version == "v1.0.0"
        assert service.status().current_model_version == "v1.0.0"

    def test_cannot_rollback_to_never_deployed(
        self, service: TrainingPipelineService
    ) -> None:
        with pytest.raises(ValueError, match="not available"):
            service.rollback(TrainingRollbackRequest(version="v9.9.9"))

    def test_versioning_bumps_patch_for_reindex(self) -> None:
        svc = TrainingPipelineService()
        svc._current_version = "v3.2.1"
        r = svc.trigger(
            TrainingTriggerRequest(type=TrainingJobType.REINDEX, affected_segments=1)
        )
        assert r.model_version == "v3.2.2"

    def test_versioning_bumps_minor_for_finetune(self) -> None:
        svc = TrainingPipelineService()
        svc._current_version = "v3.2.1"
        r = svc.trigger(
            TrainingTriggerRequest(type=TrainingJobType.FINETUNE, training_images=80)
        )
        assert r.model_version == "v3.3.0"

    def test_versioning_bumps_minor_for_full_retrain(self) -> None:
        svc = TrainingPipelineService()
        svc._current_version = "v3.2.1"
        r = svc.trigger(
            TrainingTriggerRequest(
                type=TrainingJobType.FULL_RETRAIN,
                training_images=200,
                affected_segments=50,
            )
        )
        assert r.model_version == "v3.3.0"

    def test_reset(self, service: TrainingPipelineService) -> None:
        service.trigger(
            TrainingTriggerRequest(type=TrainingJobType.FINETUNE, training_images=80)
        )
        assert len(service._jobs) == 1
        service.reset()
        assert len(service._jobs) == 0
        assert service.status().current_model_version == "v1.0.0"

    def test_list_jobs_pagination_offset(
        self, service: TrainingPipelineService
    ) -> None:
        for _ in range(5):
            service.trigger(
                TrainingTriggerRequest(
                    type=TrainingJobType.REINDEX, affected_segments=1
                )
            )
        page = service.list_jobs(offset=2, limit=2)
        assert page.total == 5
        assert len(page.items) == 2


class TestTrainingEndpoints:
    def test_get_status_initial(self, default_client: TestClient) -> None:
        response = default_client.get("/api/v1/training/status")
        assert response.status_code == 200
        body = response.json()
        assert body["current_model_version"] == "v1.0.0"

    def test_trigger_reindex(self, default_client: TestClient) -> None:
        response = default_client.post(
            "/api/v1/training/trigger",
            json={"type": "reindex", "affected_segments": 10, "archived_segments": 1},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "completed"
        assert body["model_version"] == "v1.0.1"

    def test_trigger_finetune(self, default_client: TestClient) -> None:
        response = default_client.post(
            "/api/v1/training/trigger",
            json={"type": "finetune", "training_images": 100},
        )
        assert response.status_code == 202
        assert response.json()["model_version"] == "v1.1.0"

    def test_list_jobs(self, default_client: TestClient) -> None:
        default_client.post(
            "/api/v1/training/trigger", json={"type": "reindex", "affected_segments": 5}
        )
        response = default_client.get("/api/v1/training/jobs")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1

    def test_get_job_by_id(self, default_client: TestClient) -> None:
        trigger = default_client.post(
            "/api/v1/training/trigger",
            json={"type": "reindex", "affected_segments": 3},
        )
        job_id = trigger.json()["job_id"]
        response = default_client.get(f"/api/v1/training/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["id"] == job_id

    def test_get_job_not_found(self, default_client: TestClient) -> None:
        fake_id = uuid4()
        response = default_client.get(f"/api/v1/training/jobs/{fake_id}")
        assert response.status_code == 404

    def test_cancel_job(self, default_client: TestClient) -> None:
        # create pending job manually
        from mycoai_retrieval_backend import training as mod

        job = TrainingJob(
            id=uuid4(),
            type=TrainingJobType.FINETUNE,
            status=TrainingJobStatus.PENDING,
            progress=TrainingProgress(
                stage=TrainingStage.PREFLIGHT, current=0, total=1
            ),
            changes_since_last={},
            model_version="v1.2.0",
            is_deployed=False,
            created_at=datetime.now(UTC),
        )
        mod.training_service._jobs[job.id] = job

        response = default_client.post(f"/api/v1/training/jobs/{job.id}/cancel")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    def test_cancel_completed_rejected(self, default_client: TestClient) -> None:
        trigger = default_client.post(
            "/api/v1/training/trigger",
            json={"type": "reindex", "affected_segments": 3},
        )
        job_id = trigger.json()["job_id"]
        response = default_client.post(f"/api/v1/training/jobs/{job_id}/cancel")
        assert response.status_code == 409

    def test_deploy_finetune_job(self, default_client: TestClient) -> None:
        trigger = default_client.post(
            "/api/v1/training/trigger",
            json={"type": "finetune", "training_images": 60},
        )
        job_id = trigger.json()["job_id"]
        response = default_client.post(f"/api/v1/training/jobs/{job_id}/deploy")
        assert response.status_code == 200
        assert response.json()["is_deployed"] is True

    def test_deploy_reindex_job_rejected(self, default_client: TestClient) -> None:
        trigger = default_client.post(
            "/api/v1/training/trigger",
            json={"type": "reindex", "affected_segments": 1},
        )
        job_id = trigger.json()["job_id"]
        response = default_client.post(f"/api/v1/training/jobs/{job_id}/deploy")
        assert response.status_code == 409

    def test_rollback(self, default_client: TestClient) -> None:
        # advance to v1.0.1
        default_client.post(
            "/api/v1/training/trigger",
            json={"type": "reindex", "affected_segments": 2},
        )
        response = default_client.post(
            "/api/v1/training/rollback",
            json={"version": "v1.0.0"},
        )
        assert response.status_code == 200
        assert response.json()["model_version"] == "v1.0.0"

    def test_rollback_invalid_version(self, default_client: TestClient) -> None:
        response = default_client.post(
            "/api/v1/training/rollback",
            json={"version": "v99.0.0"},
        )
        assert response.status_code == 409
