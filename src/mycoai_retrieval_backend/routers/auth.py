from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..database import get_db
from ..deps import get_current_user
from ..models import User
from ..repo import (
    refresh_token_find,
    refresh_token_revoke,
    refresh_token_save,
    user_by_email,
    user_count,
    user_create,
)
from ..schemas import (
    AuthLoginRequest,
    AuthRefreshRequest,
    AuthRegisterRequest,
    AuthTokenResponse,
    UserResponse,
)
from ..security import create_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])
limiter = Limiter(key_func=get_remote_address)


def _set_refresh_cookie(response: Response, refresh_token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=settings.environment != "development",
        samesite="strict",
        max_age=int(settings.refresh_token_lifetime.total_seconds()),
        path="/api/v1/auth",
    )


@router.post(
    "/register", response_model=AuthTokenResponse, status_code=status.HTTP_201_CREATED
)
@limiter.limit("5/minute")
async def register(
    request: Request,
    body: AuthRegisterRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> AuthTokenResponse:
    existing = await user_by_email(db, body.email)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Email already registered"
        )

    total = await user_count(db)
    role = "owner" if total == 0 else "user"

    password_hash = hash_password(body.password)
    user = await user_create(
        db, email=body.email, password_hash=password_hash, name=body.name, role=role
    )
    await db.commit()

    settings = get_settings()
    access_token = create_token(
        user.id, "access", int(settings.access_token_lifetime.total_seconds())
    )
    refresh_token = create_token(
        user.id, "refresh", int(settings.refresh_token_lifetime.total_seconds())
    )
    expires_at = datetime.now(tz=UTC) + settings.refresh_token_lifetime
    await refresh_token_save(db, user.id, refresh_token, expires_at)
    await db.commit()

    _set_refresh_cookie(response, refresh_token)
    return AuthTokenResponse(
        access_token=access_token, refresh_token=refresh_token, token_type="bearer"
    )


@router.post("/login", response_model=AuthTokenResponse)
@limiter.limit("5/minute")
async def login(
    request: Request,
    body: AuthLoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> AuthTokenResponse:
    user = await user_by_email(db, body.email)
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )

    settings = get_settings()
    access_token = create_token(
        user.id, "access", int(settings.access_token_lifetime.total_seconds())
    )
    refresh_token = create_token(
        user.id, "refresh", int(settings.refresh_token_lifetime.total_seconds())
    )
    expires_at = datetime.now(tz=UTC) + settings.refresh_token_lifetime
    await refresh_token_save(db, user.id, refresh_token, expires_at)
    await db.commit()

    _set_refresh_cookie(response, refresh_token)
    return AuthTokenResponse(
        access_token=access_token, refresh_token=refresh_token, token_type="bearer"
    )


@router.post("/refresh", response_model=AuthTokenResponse)
async def refresh(
    request: Request,
    body: AuthRefreshRequest | None = None,
    db: AsyncSession = Depends(get_db),
) -> AuthTokenResponse:
    refresh_token_str: str | None = None
    if body is not None:
        refresh_token_str = body.refresh_token
    if refresh_token_str is None:
        refresh_token_str = request.cookies.get("refresh_token")
    if refresh_token_str is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing refresh token"
        )

    stored = await refresh_token_find(db, refresh_token_str)
    if (
        stored is None
        or stored.revoked_at is not None
        or stored.expires_at.replace(tzinfo=UTC) < datetime.now(tz=UTC)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )

    settings = get_settings()
    access_token = create_token(
        stored.user_id, "access", int(settings.access_token_lifetime.total_seconds())
    )
    return AuthTokenResponse(access_token=access_token, token_type="bearer")


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    body: AuthRefreshRequest | None = None,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    refresh_token_str: str | None = None
    if body is not None:
        refresh_token_str = body.refresh_token
    if refresh_token_str is None:
        refresh_token_str = request.cookies.get("refresh_token")
    if refresh_token_str is not None:
        await refresh_token_revoke(db, refresh_token_str)
        await db.commit()
    response.delete_cookie(key="refresh_token", path="/api/v1/auth")


@router.get("/me", response_model=UserResponse)
async def me(user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role,
        is_active=user.is_active,
    )
