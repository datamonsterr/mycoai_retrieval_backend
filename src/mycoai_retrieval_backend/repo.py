from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import RefreshToken, User


async def user_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def user_by_id(db: AsyncSession, user_id: str) -> User | None:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def user_count(db: AsyncSession) -> int:
    result = await db.execute(select(User))
    return len(result.scalars().all())


async def user_create(
    db: AsyncSession, email: str, password_hash: str, name: str, role: str
) -> User:
    user = User(email=email, password_hash=password_hash, name=name, role=role)
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


async def refresh_token_save(
    db: AsyncSession, user_id: str, token: str, expires_at: datetime
) -> RefreshToken:
    rt = RefreshToken(token=token, user_id=user_id, expires_at=expires_at)
    db.add(rt)
    return rt


async def refresh_token_find(db: AsyncSession, token: str) -> RefreshToken | None:
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.token == token, RefreshToken.revoked_at.is_(None)
        )
    )
    return result.scalar_one_or_none()


async def refresh_token_revoke(db: AsyncSession, token: str) -> None:
    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.token == token)
        .values(revoked_at=datetime.now(tz=UTC))
    )
