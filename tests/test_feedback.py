from uuid import UUID

from fastapi.testclient import TestClient

from mycoai_retrieval_backend.app import app
from mycoai_retrieval_backend.feedback_store import reset_store

client = TestClient(app)


def setup_function() -> None:
    reset_store()


def test_feedback_submission_and_review_flow() -> None:
    submitter_id = str(UUID(int=1))
    reviewer_id = str(UUID(int=2))

    created = client.post(
        "/api/v1/feedback",
        json={
            "submitter_id": submitter_id,
            "source": "query_result",
            "query_strain": "strain-a",
            "predicted_species": "species-a",
            "suggested_species": "species-b",
            "description": "wrong species",
        },
    )
    assert created.status_code == 201
    feedback_id = created.json()["feedback"]["id"]

    inbox = client.get("/api/v1/feedback/inbox")
    assert inbox.status_code == 200
    assert inbox.json()["feedback"][0]["id"] == feedback_id

    reviewed = client.patch(
        f"/api/v1/feedback/{feedback_id}",
        json={
            "reviewer_id": reviewer_id,
            "status": "accepted",
            "review_note": "fixed",
        },
    )
    assert reviewed.status_code == 200
    assert reviewed.json()["feedback"]["status"] == "accepted"

    listed = client.get("/api/v1/feedback", params={"submitter_id": submitter_id})
    assert listed.status_code == 200
    assert listed.json()["feedback"][0]["review_note"] == "fixed"

    stats = client.get("/api/v1/feedback/stats")
    assert stats.status_code == 200
    assert stats.json()["stats"]["acceptance_rate"] == 1.0


def test_reject_requires_review_note() -> None:
    submitter_id = str(UUID(int=3))

    created = client.post(
        "/api/v1/feedback",
        json={
            "submitter_id": submitter_id,
            "source": "database_review",
            "image_id": str(UUID(int=4)),
            "suggested_species": "species-c",
            "description": "database issue",
        },
    )
    feedback_id = created.json()["feedback"]["id"]

    response = client.patch(
        f"/api/v1/feedback/{feedback_id}",
        json={
            "reviewer_id": str(UUID(int=5)),
            "status": "rejected",
        },
    )
    assert response.status_code == 422
