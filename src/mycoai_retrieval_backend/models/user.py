from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, EmailStr, Field

AUDIT_ACTION_PROMOTE = "promote_to_data_owner"
AUDIT_ACTION_DEMOTE = "demote_to_normal_user"


class UserRole(StrEnum):
    NORMAL = "normal_user"
    DATA_OWNER = "data_owner"


class UserCreate(BaseModel):
    email: EmailStr
    password: Annotated[str, Field(min_length=8)]
    name: Annotated[str, Field(min_length=1, max_length=200)]


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserUpdate(BaseModel):
    role: UserRole


class User(BaseModel):
    id: int
    email: EmailStr
    name: str
    role: UserRole
    is_active: bool = True


class UserInDB(User):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AuditEntry(BaseModel):
    id: int
    actor_id: int
    target_id: int
    action: str
    timestamp: float
