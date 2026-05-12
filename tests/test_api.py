from fastapi.testclient import TestClient

from mycoai_retrieval_backend.app import create_app

client = TestClient(create_app())


def test_router_aggregates_all_endpoints() -> None:
    resp = client.get("/api/v1/dashboard/stats")
    assert resp.status_code == 200
    assert resp.json()["images"] == 0


def test_auth_login_returns_token() -> None:
    resp = client.post("/api/v1/auth/login")
    assert resp.status_code == 200
    assert resp.json()["token_type"] == "bearer"


def test_images_upload_returns_accepted() -> None:
    resp = client.post(
        "/api/v1/images/upload", files={"file": ("test.jpg", b"data", "image/jpeg")}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"


def test_retrieval_query_returns_empty() -> None:
    resp = client.post("/api/v1/retrieval/query", json={"query": "test"})
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_species_crud_works() -> None:
    resp = client.get("/api/v1/species")
    assert resp.status_code == 200
    assert resp.json() == []


def test_feedback_create_works() -> None:
    resp = client.post(
        "/api/v1/feedback", json={"image_id": "img-1", "label": "correct"}
    )
    assert resp.status_code == 200
    assert resp.json()["label"] == "correct"


def test_training_trigger_works() -> None:
    resp = client.post("/api/v1/training/trigger", json={"reason": "new data"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"
