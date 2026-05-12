from dataclasses import dataclass
from typing import Literal

RoleName = Literal["admin", "scientist", "viewer"]


@dataclass(frozen=True)
class User:
    id: str
    email: str
    roles: tuple[RoleName, ...]
    service: str


def require_role(user: User, role: RoleName) -> None:
    if role not in user.roles:
        msg = f"Missing required role: {role}"
        raise PermissionError(msg)
