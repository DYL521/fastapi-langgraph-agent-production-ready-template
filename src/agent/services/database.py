"""Database connection management.

Owns the SQLAlchemy engine and connection pool plus health checks. Entity CRUD
lives in the repositories under ``agent.repositories``.
"""

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool
from sqlmodel import (
    Session,
    create_engine,
    select,
)

from agent.core.config import (
    Environment,
    settings,
)
from agent.core.db import build_sqlalchemy_url
from agent.core.logging import logger


class Database:
    """Owns the engine/connection pool and exposes session + health helpers."""

    def __init__(self):
        """Initialize the engine with a dialect-appropriate connection pool."""
        try:
            connection_url = build_sqlalchemy_url(async_=False)
            self.engine = create_engine(
                connection_url,
                pool_pre_ping=True,
                poolclass=QueuePool,
                pool_size=settings.database.pool_size,
                max_overflow=settings.database.max_overflow,
                pool_timeout=30,  # Connection timeout (seconds)
                pool_recycle=1800,  # Recycle connections after 30 minutes
            )
            logger.info(
                "database_initialized",
                environment=settings.app.environment.value,
                pool_size=settings.database.pool_size,
                max_overflow=settings.database.max_overflow,
            )
        except SQLAlchemyError:
            logger.exception("database_initialization_error", environment=settings.app.environment.value)
            # In production, don't raise - allow app to start even with DB issues
            if settings.app.environment != Environment.PRODUCTION:
                raise

    def get_session_maker(self) -> Session:
        """Return a new SQLModel session bound to the engine."""
        return Session(self.engine)

    async def health_check(self) -> bool:
        """Check database connection health.

        Returns:
            bool: True if the database is reachable, False otherwise.
        """
        try:
            with Session(self.engine) as session:
                session.exec(select(1)).first()
                return True
        except Exception:
            logger.exception("database_health_check_failed")
            return False


# Singleton instance — owns the shared engine/connection pool.
database = Database()
