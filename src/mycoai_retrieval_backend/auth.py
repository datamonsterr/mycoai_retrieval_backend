from __future__ import annotations

import base64
import hashlib
import hmac

from .config import get_settings
from .models.user import Token, User


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_access_token(user: User) -> Token:
    settings = get_settings()
    payload = f"{user.id}:{user.email}"
    signature = hmac.new(
        settings.jwt_secret.encode(), payload.encode(), hashlib.sha256
    ).digest()
    token = f"{_b64encode(payload.encode())}.{_b64encode(signature)}"
    return Token(access_token=token)


def decode_user_id(token: str) -> int:
    settings = get_settings()
    try:
        payload_b64, sig_b64 = token.split(".", 1)
        payload = _b64decode(payload_b64).decode()
        signature = _b64decode(sig_b64)
    except ValueError as exc:
        raise ValueError("Invalid token") from exc

    expected = hmac.new(
        settings.jwt_secret.encode(), payload.encode(), hashlib.sha256
    ).digest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("Invalid token")

    user_id_str, _, _ = payload.partition(":")
    if not user_id_str.isdigit():
        raise ValueError("Invalid token")
    return int(user_id_str)
