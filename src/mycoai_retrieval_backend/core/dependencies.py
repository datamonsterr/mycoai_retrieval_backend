from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends

from mycoai_retrieval_backend.config import Settings, get_settings
from mycoai_retrieval_backend.models.user import User

SettingsDep = Annotated[Settings, Depends(get_settings)]


async def get_db() -> AsyncIterator[None]:
    yield None


async def get_current_user(settings: SettingsDep) -> User:
    return User(
        id="development",
        email="dev@myco.ai",
        roles=("admin",),
        service=settings.app_name,
    )
