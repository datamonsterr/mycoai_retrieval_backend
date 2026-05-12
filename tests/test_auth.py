from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from fastapi import Depends
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from mycoai_retrieval_backend.app import create_app
from mycoai_retrieval_backend.database import Base, get_db
from mycoai_retrieval_backend.deps import require_role


@pytest.fixture
async def client(tmp_path: Path) -> AsyncGenerator[AsyncClient]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    engine = create_async_engine(
        database_url, connect_args={"check_same_thread": False}
    )
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async def override_get_db() -> AsyncGenerator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app = create_app(create_tables=False)
    app.state.limiter.enabled = False
    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as async_client:
        yield async_client

    await engine.dispose()


@pytest.mark.anyio
async def test_first_registration_creates_owner_and_returns_tokens(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "owner@example.com", "password": "password123", "name": "Owner"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["token_type"] == "bearer"
    assert "refresh_token" in response.cookies

    me = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert me.status_code == 200
    assert me.json()["role"] == "owner"


@pytest.mark.anyio
async def test_second_registration_creates_user(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/auth/register",
        json={"email": "owner@example.com", "password": "password123", "name": "Owner"},
    )

    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "user@example.com", "password": "password123", "name": "User"},
    )

    assert response.status_code == 201
    token = response.json()["access_token"]
    me = await client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
    )
    assert me.json()["role"] == "user"


@pytest.mark.anyio
async def test_password_min_length_enforced(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "bad@example.com", "password": "short", "name": "Bad"},
    )

    assert response.status_code == 422


@pytest.mark.anyio
async def test_login_refresh_and_logout_revoke_refresh_token(
    client: AsyncClient,
) -> None:
    await client.post(
        "/api/v1/auth/register",
        json={"email": "owner@example.com", "password": "password123", "name": "Owner"},
    )

    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "owner@example.com", "password": "password123"},
    )
    assert login.status_code == 200
    tokens = login.json()

    refreshed = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert refreshed.status_code == 200
    assert refreshed.json()["access_token"]

    logout = await client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": tokens["refresh_token"]},
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )
    assert logout.status_code == 204

    refreshed_after_logout = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert refreshed_after_logout.status_code == 401


@pytest.mark.anyio
async def test_invalid_login_rejected(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/auth/register",
        json={"email": "owner@example.com", "password": "password123", "name": "Owner"},
    )

    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "owner@example.com", "password": "wrongpassword"},
    )

    assert response.status_code == 401


@pytest.mark.anyio
async def test_require_role_blocks_non_owner(client: AsyncClient) -> None:
    app = client._transport.app

    @app.get("/owner-only")
    async def owner_only(user=Depends(require_role("owner"))):
        return {"id": user.id}

    await client.post(
        "/api/v1/auth/register",
        json={"email": "owner@example.com", "password": "password123", "name": "Owner"},
    )
    user = await client.post(
        "/api/v1/auth/register",
        json={"email": "user@example.com", "password": "password123", "name": "User"},
    )

    response = await client.get(
        "/owner-only",
        headers={"Authorization": f"Bearer {user.json()['access_token']}"},
    )

    assert response.status_code == 403


@pytest.mark.anyio
async def test_inactive_user_rejected(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/auth/register",
        json={"email": "owner@example.com", "password": "password123", "name": "Owner"},
    )

    response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer bad-token"},
    )
    assert response.status_code == 401

    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 401
