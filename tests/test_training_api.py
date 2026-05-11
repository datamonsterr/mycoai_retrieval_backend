from fastapi.testclient import TestClient

from mycoai_retrieval_backend.app import app

client = TestClient(app)


def test_training_status_exposes_model_and_history() -> None:
    response = client.get("/api/v1/training/status")

    assert response.status_code == 200
    body = response.json()
    assert body["current_model"]["version"] == "v3.2.1"
    assert body["current_model"]["staged_version"] == "v3.3.0"
    assert body["latest_job"]["job_id"] == "train_20260506_001"


def test_training_jobs_returns_history() -> None:
    response = client.get("/api/v1/training/jobs")

    assert response.status_code == 200
    jobs = response.json()
    assert len(jobs) >= 1
    assert jobs[0]["status"] == "completed"


def test_training_preflight_shows_summary() -> None:
    response = client.get(
        "/api/v1/training/preflight",
        params={"job_type": "full_retrain"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job_type"] == "full_retrain"
    assert body["changes_since_last"] == {
        "strains_added": 12,
        "strains_archived": 3,
        "feedback_accepted": 5,
    }
    assert body["can_start"] is True


def test_trigger_and_cancel_training_job() -> None:
    trigger_response = client.post(
        "/api/v1/training/trigger",
        json={"job_type": "reindex", "trigger": "manual"},
    )

    assert trigger_response.status_code == 202
    job = trigger_response.json()
    assert job["status"] == "running"
    assert job["progress"]["stage"] == "preparing"

    cancel_response = client.post(f"/api/v1/training/jobs/{job['job_id']}/cancel")

    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "cancelled"


def test_deploy_and_rollback_training_model() -> None:
    deploy_response = client.post("/api/v1/training/jobs/train_20260506_001/deploy")

    assert deploy_response.status_code == 200
    assert deploy_response.json()["deployment_status"] == "deployed"

    rollback_response = client.post(
        "/api/v1/training/rollback",
        json={"version": "v3.2.1"},
    )

    assert rollback_response.status_code == 202
    assert rollback_response.json()["deployment_status"] == "rolled_back"
