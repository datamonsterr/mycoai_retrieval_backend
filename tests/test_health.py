from fastapi.testclient import TestClient

from mycoai_retrieval_backend.app import app
from mycoai_retrieval_backend.feedback import _store

client = TestClient(app)


def setup_function() -> None:
    _store.clear()


def test_healthcheck_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_exposes_key_endpoints() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "name": "MycoAI Retrieval Backend",
        "docs": "/docs",
        "health": "/health",
    }


def test_feedback_flow_supports_submit_list_and_review() -> None:
    created = client.post(
        "/api/v1/feedback/",
        json={
            "source": "query_result",
            "query_strain": "strain-12",
            "predicted_species": "Species A",
            "suggested_species": "Species B",
            "description": "prediction looks wrong",
            "submitter_id": "user-1",
        },
    )

    assert created.status_code == 201
    payload = created.json()
    assert payload["status"] == "pending"
    assert payload["source"] == "query_result"

    listed = client.get("/api/v1/feedback/", params={"status": "pending"})
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    feedback_id = payload["feedback_id"]
    reviewed = client.patch(
        f"/api/v1/feedback/{feedback_id}/review",
        json={
            "action": "accept",
            "reviewed_by": "owner-1",
            "review_note": "confirmed",
        },
    )

    assert reviewed.status_code == 200
    assert reviewed.json()["status"] == "accepted"
    assert reviewed.json()["reviewed_by"] == "owner-1"


def test_feedback_bulk_review_skips_non_pending_items() -> None:
    first = client.post(
        "/api/v1/feedback/",
        json={
            "source": "database_review",
            "query_strain": "strain-1",
            "predicted_species": "Species A",
            "suggested_species": "Species C",
            "description": "bad label",
            "submitter_id": "user-2",
        },
    ).json()["feedback_id"]
    second = client.post(
        "/api/v1/feedback/",
        json={
            "source": "database_review",
            "query_strain": "strain-2",
            "predicted_species": "Species A",
            "suggested_species": "Species C",
            "description": "bad label",
            "submitter_id": "user-2",
        },
    ).json()["feedback_id"]

    response = client.patch(
        "/api/v1/feedback/review/bulk",
        json={
            "action": "reject",
            "reviewed_by": "owner-9",
            "review_note": "duplicate",
            "feedback_ids": [first, second],
        },
    )

    assert response.status_code == 200
    assert {item["status"] for item in response.json()} == {"rejected"}
