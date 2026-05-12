from typing import Any, cast

from fastapi.testclient import TestClient

PREFIX = "/api/auth"


def _register(
    client: TestClient,
    email: str,
    password: str = "password123",
    name: str = "User",
) -> dict[str, Any]:
    response = client.post(
        f"{PREFIX}/register",
        json={"email": email, "password": password, "name": name},
    )
    assert response.status_code == 201
    return cast(dict[str, Any], response.json())


def _login(
    client: TestClient, email: str, password: str = "password123"
) -> dict[str, str]:
    response = client.post(
        f"{PREFIX}/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_first_registered_user_is_data_owner(client: TestClient) -> None:
    user = _register(client, "owner@test.com", name="Owner")

    assert user["role"] == "data_owner"
    assert user["is_active"] is True
    assert "password" not in user
    assert "hashed_password" not in user


def test_second_registered_user_is_normal_user(client: TestClient) -> None:
    _register(client, "owner@test.com", name="Owner")
    user = _register(client, "normal@test.com", name="Normal")

    assert user["role"] == "normal_user"


def test_registration_rejects_duplicate_email(client: TestClient) -> None:
    _register(client, "owner@test.com", name="Owner")

    response = client.post(
        f"{PREFIX}/register",
        json={"email": "owner@test.com", "password": "password123", "name": "Dup"},
    )

    assert response.status_code == 409


def test_registration_validates_email_password_and_name(client: TestClient) -> None:
    invalid_cases = [
        {"email": "not-email", "password": "password123", "name": "Name"},
        {"email": "short@test.com", "password": "short", "name": "Name"},
        {"email": "name@test.com", "password": "password123", "name": ""},
    ]

    for payload in invalid_cases:
        response = client.post(f"{PREFIX}/register", json=payload)
        assert response.status_code == 422


def test_login_returns_jwt_bearer_token(client: TestClient) -> None:
    _register(client, "owner@test.com", name="Owner")

    response = client.post(
        f"{PREFIX}/login",
        json={"email": "owner@test.com", "password": "password123"},
    )

    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"
    assert response.json()["access_token"]


def test_login_rejects_bad_credentials(client: TestClient) -> None:
    _register(client, "owner@test.com", name="Owner")

    bad_password = client.post(
        f"{PREFIX}/login",
        json={"email": "owner@test.com", "password": "wrong"},
    )
    missing_user = client.post(
        f"{PREFIX}/login",
        json={"email": "none@test.com", "password": "password123"},
    )

    assert bad_password.status_code == 401
    assert missing_user.status_code == 401


def test_profile_requires_jwt_and_returns_current_user(client: TestClient) -> None:
    _register(client, "owner@test.com", name="Owner")
    headers = _login(client, "owner@test.com")

    missing = client.get(f"{PREFIX}/me")
    profile = client.get(f"{PREFIX}/me", headers=headers)

    assert missing.status_code == 401
    assert profile.status_code == 200
    assert profile.json()["email"] == "owner@test.com"
    assert profile.json()["role"] == "data_owner"


def test_invalid_token_rejected(client: TestClient) -> None:
    response = client.get(
        f"{PREFIX}/me",
        headers={"Authorization": "Bearer garbage.token.here"},
    )

    assert response.status_code == 401


def test_normal_user_gets_403_for_data_owner_endpoints(
    client: TestClient,
    normal_user: dict[str, str],
) -> None:
    response = client.get(f"{PREFIX}/users", headers=normal_user)

    assert response.status_code == 403
    assert "Data Owner role required" in response.json()["detail"]


def test_unauthenticated_user_gets_401_for_data_owner_endpoints(
    client: TestClient,
) -> None:
    response = client.get(f"{PREFIX}/users")

    assert response.status_code == 401


def test_data_owner_can_manage_user_roles(
    client: TestClient,
    data_owner: dict[str, str],
) -> None:
    normal = _register(client, "normal@test.com", name="Normal")

    promoted = client.patch(
        f"{PREFIX}/users/{normal['id']}",
        json={"role": "data_owner"},
        headers=data_owner,
    )
    demoted = client.patch(
        f"{PREFIX}/users/{normal['id']}",
        json={"role": "normal_user"},
        headers=data_owner,
    )

    assert promoted.status_code == 200
    assert promoted.json()["role"] == "data_owner"
    assert demoted.status_code == 200
    assert demoted.json()["role"] == "normal_user"


def test_role_changes_are_audited(
    client: TestClient,
    data_owner: dict[str, str],
) -> None:
    normal = _register(client, "normal@test.com", name="Normal")

    client.patch(
        f"{PREFIX}/users/{normal['id']}",
        json={"role": "data_owner"},
        headers=data_owner,
    )
    response = client.get(f"{PREFIX}/audit", headers=data_owner)

    assert response.status_code == 200
    entries = response.json()
    assert len(entries) == 1
    assert entries[0]["target_id"] == normal["id"]
    assert entries[0]["action"] == "promote_to_data_owner"


def test_audit_log_requires_data_owner(
    client: TestClient,
    normal_user: dict[str, str],
) -> None:
    response = client.get(f"{PREFIX}/audit", headers=normal_user)

    assert response.status_code == 403


def test_data_owner_cannot_demote_self(
    client: TestClient,
    data_owner: dict[str, str],
) -> None:
    owner = client.get(f"{PREFIX}/me", headers=data_owner).json()

    response = client.patch(
        f"{PREFIX}/users/{owner['id']}",
        json={"role": "normal_user"},
        headers=data_owner,
    )

    assert response.status_code == 403
    assert "own role" in response.json()["detail"]


def test_role_management_rejects_missing_user(
    client: TestClient,
    data_owner: dict[str, str],
) -> None:
    response = client.patch(
        f"{PREFIX}/users/999",
        json={"role": "normal_user"},
        headers=data_owner,
    )

    assert response.status_code == 404
