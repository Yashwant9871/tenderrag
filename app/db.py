# app/db.py
import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from app.config import settings
from app.models import Base

logger = logging.getLogger(__name__)

# Engine: tune pool size via env/config (SQLAlchemy will pass through to asyncpg)
engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
    # Optional tuning:
    # pool_size=settings.db_pool_size,
    # max_overflow=settings.db_max_overflow,
)

# Use async_sessionmaker (SQLAlchemy 1.4+/2.0 style for async)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_models() -> None:
    """
    Development helper that creates tables from ORM metadata.
    In production, prefer Alembic migrations instead of create_all().
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/checked")


async def close_engine() -> None:
    """Call this on app shutdown to cleanly dispose connection pool."""
    await engine.dispose()
    logger.info("Database engine disposed")


# FastAPI dependency
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Use in FastAPI routes like:
        async def endpoint(session: AsyncSession = Depends(get_async_session)):
            ...
    Ensures session is closed and rolled back on error.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
