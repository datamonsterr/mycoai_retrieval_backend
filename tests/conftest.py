import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mycoai_retrieval_backend.app import create_app
from mycoai_retrieval_backend.deps import get_user_store
from mycoai_retrieval_backend.services.user_store import UserStore


@pytest.fixture
def app_with_fresh_store() -> FastAPI:
    app = create_app()
    store = UserStore()
    app.dependency_overrides[get_user_store] = lambda: store
    return app


@pytest.fixture
def client(app_with_fresh_store: FastAPI) -> TestClient:
    return TestClient(app_with_fresh_store)


@pytest.fixture
def data_owner(client: TestClient) -> dict[str, str]:
    resp = client.post(
        "/api/auth/register",
        json={"email": "owner@test.com", "password": "password123", "name": "Owner"},
    )
    assert resp.status_code == 201
    login = client.post(
        "/api/auth/login",
        json={"email": "owner@test.com", "password": "password123"},
    )
    assert login.status_code == 200
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest.fixture
def normal_user(client: TestClient) -> dict[str, str]:
    client.post(
        "/api/auth/register",
        json={"email": "owner@test.com", "password": "password123", "name": "Owner"},
    )
    client.post(
        "/api/auth/register",
        json={"email": "norm@test.com", "password": "password123", "name": "Norm"},
    )
    login = client.post(
        "/api/auth/login",
        json={"email": "norm@test.com", "password": "password123"},
    )
    assert login.status_code == 200
    return {"Authorization": f"Bearer {login.json()['access_token']}"}
