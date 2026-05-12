import hashlib
import hmac
import secrets
from collections.abc import Sequence

from ..models.user import (
    AUDIT_ACTION_DEMOTE,
    AUDIT_ACTION_PROMOTE,
    AuditEntry,
    User,
    UserCreate,
    UserInDB,
    UserRole,
    UserUpdate,
)


class ConflictError(ValueError):
    pass


class ForbiddenError(ValueError):
    pass


class NotFoundError(ValueError):
    pass


class UserStore:
    def __init__(self) -> None:
        self._users: dict[int, UserInDB] = {}
        self._email_index: dict[str, int] = {}
        self._next_id: int = 1
        self._audit: list[AuditEntry] = []
        self._audit_next_id: int = 1

    def register(self, payload: UserCreate, auto_activate: bool = True) -> User:
        if payload.email in self._email_index:
            raise ConflictError(f"Email {payload.email} already registered")

        uid = self._next_id
        self._next_id += 1

        role = UserRole.DATA_OWNER if len(self._users) == 0 else UserRole.NORMAL

        user = UserInDB(
            id=uid,
            email=payload.email,
            name=payload.name,
            role=role,
            is_active=auto_activate,
            hashed_password=_hash_password(payload.password),
        )
        self._users[uid] = user
        self._email_index[payload.email] = uid
        return self._to_public(user)

    def authenticate(self, email: str, password: str) -> User:
        uid = self._email_index.get(email)
        if uid is None:
            raise NotFoundError("Invalid email or password")
        user = self._users[uid]
        if not user.is_active:
            raise ForbiddenError("Account is not activated")
        if not _verify_password(password, user.hashed_password):
            raise NotFoundError("Invalid email or password")
        return self._to_public(user)

    def get_by_id(self, user_id: int) -> User:
        user = self._users.get(user_id)
        if user is None:
            raise NotFoundError(f"User {user_id} not found")
        return self._to_public(user)

    def update_role(self, actor_id: int, target_id: int, payload: UserUpdate) -> User:
        if target_id not in self._users:
            raise NotFoundError(f"User {target_id} not found")
        if actor_id == target_id:
            raise ForbiddenError("Cannot change your own role")

        target = self._users[target_id]
        new_role = payload.role

        if target.role == new_role:
            return self._to_public(target)

        if new_role == UserRole.DATA_OWNER:
            target.role = UserRole.DATA_OWNER
            self._add_audit(actor_id, target_id, AUDIT_ACTION_PROMOTE)
        else:
            owners = [u for u in self._users.values() if u.role == UserRole.DATA_OWNER]
            if len(owners) <= 1:
                raise ForbiddenError("At least one Data Owner must exist")
            target.role = UserRole.NORMAL
            self._add_audit(actor_id, target_id, AUDIT_ACTION_DEMOTE)

        return self._to_public(target)

    def list_users(self) -> list[User]:
        return [self._to_public(u) for u in self._users.values()]

    def get_audit_log(self) -> Sequence[AuditEntry]:
        return list(self._audit)

    def _add_audit(self, actor_id: int, target_id: int, action: str) -> None:
        entry = AuditEntry(
            id=self._audit_next_id,
            actor_id=actor_id,
            target_id=target_id,
            action=action,
            timestamp=0.0,
        )
        self._audit_next_id += 1
        self._audit.append(entry)

    @staticmethod
    def _to_public(user: UserInDB) -> User:
        return User(
            id=user.id,
            email=user.email,
            name=user.name,
            role=user.role,
            is_active=user.is_active,
        )


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 200_000
    ).hex()
    return f"{salt}${digest}"


def _verify_password(password: str, stored: str) -> bool:
    salt, digest = stored.split("$", 1)
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 200_000
    ).hex()
    return hmac.compare_digest(candidate, digest)
