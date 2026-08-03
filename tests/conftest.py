"""Shared fixtures for integration tests.

Uses an in-memory async SQLite database (StaticPool so every session shares the
one connection) and injects it into app.state.database, so the FastAPI dependency
injection resolves repositories against the test DB automatically.
"""

from types import SimpleNamespace

import pytest_asyncio
from httpx import (
    ASGITransport,
    AsyncClient,
)
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

# Import models so their tables are registered on SQLModel.metadata.
import agent.models.session  # noqa: F401
import agent.models.thread  # noqa: F401
import agent.models.user  # noqa: F401


@pytest_asyncio.fixture
async def session_maker():
    """Provide an async session factory over a fresh in-memory SQLite DB."""
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield maker
    await engine.dispose()


@pytest_asyncio.fixture
async def client(session_maker, monkeypatch):
    """Provide an httpx client wired to the app with the SQLite-backed repos."""
    from agent.core.limiter import limiter

    monkeypatch.setattr(limiter, "enabled", False)

    from agent.main import app

    app.state.database = SimpleNamespace(session_maker=session_maker)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
