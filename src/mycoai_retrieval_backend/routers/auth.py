from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import create_access_token
from ..deps import get_current_user, get_user_store, require_data_owner
from ..models.user import AuditEntry, User, UserCreate, UserLogin, UserUpdate
from ..services.user_store import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    UserStore,
)
from .tags import AUTH

router = APIRouter(prefix="/auth", tags=[AUTH])


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
def register(
    payload: UserCreate,
    store: Annotated[UserStore, Depends(get_user_store)],
) -> User:
    try:
        return store.register(payload)
    except ConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.post("/login", response_model=dict[str, str])
def login(
    payload: UserLogin,
    store: Annotated[UserStore, Depends(get_user_store)],
) -> dict[str, str]:
    try:
        user = store.authenticate(payload.email, payload.password)
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        ) from exc
    except ForbiddenError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)
        ) from exc
    token = create_access_token(user)
    return {
        "access_token": token.access_token,
        "token_type": token.token_type,
    }


@router.get("/me", response_model=User)
def get_profile(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    return current_user


@router.get("/users", response_model=list[User])
def list_users(
    _: Annotated[User, Depends(require_data_owner)],
    store: Annotated[UserStore, Depends(get_user_store)],
) -> list[User]:
    return store.list_users()


@router.patch("/users/{user_id}", response_model=User)
def update_user_role(
    user_id: int,
    payload: UserUpdate,
    current_user: Annotated[User, Depends(require_data_owner)],
    store: Annotated[UserStore, Depends(get_user_store)],
) -> User:
    try:
        return store.update_role(current_user.id, user_id, payload)
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except ForbiddenError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)
        ) from exc


@router.get("/audit", response_model=list[AuditEntry])
def get_audit_log(
    _: Annotated[User, Depends(require_data_owner)],
    store: Annotated[UserStore, Depends(get_user_store)],
) -> Sequence[AuditEntry]:
    return store.get_audit_log()
