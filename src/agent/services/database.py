"""Database connection management (async).

Owns the async SQLAlchemy engine and session factory plus health checks. Entity
CRUD lives in the repositories under ``agent.repositories``.
"""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from agent.core.config import settings
from agent.core.db import build_sqlalchemy_url
from agent.core.logging import logger


class Database:
    """Owns the async engine/connection pool and exposes a session factory."""

    def __init__(self):
        """Create the async engine and session factory (no I/O until first use)."""
        # pool_pre_ping is incompatible with the async MySQL drivers
        # (asyncmy/aiomysql ping() signature mismatch in SQLAlchemy); pool_recycle
        # still guards against stale connections. Postgres async keeps pre-ping.
        self.engine: AsyncEngine = create_async_engine(
            build_sqlalchemy_url(async_=True),
            pool_pre_ping=settings.database.dialect != "mysql",
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_timeout=30,  # Connection acquire timeout (seconds)
            pool_recycle=1800,  # Recycle connections after 30 minutes
        )
        # expire_on_commit=False so attributes stay readable after commit without
        # a lazy (and in async, illegal) refresh round-trip.
        self.session_maker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        logger.info(
            "database_initialized",
            environment=settings.app.environment.value,
            dialect=settings.database.dialect,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
        )

    async def health_check(self) -> bool:
        """Check database connection health.

        Returns:
            bool: True if the database is reachable, False otherwise.
        """
        try:
            async with self.session_maker() as session:
                await session.exec(select(1))
                return True
        except Exception:
            logger.exception("database_health_check_failed")
            return False

    async def dispose(self) -> None:
        """Dispose of the engine's connection pool (call on shutdown)."""
        await self.engine.dispose()
        logger.info("database_disposed")
