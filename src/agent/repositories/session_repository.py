"""Chat session persistence (CRUD) backed by async SQLModel."""

from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import (
    col,
    select,
    update,
)
from sqlmodel.ext.asyncio.session import AsyncSession

from agent.core.exceptions import NotFoundError
from agent.core.logging import logger
from agent.models.session import Session as ChatSession
from agent.services.database import database


class SessionRepository:
    """CRUD operations for chat ``Session`` records."""

    def __init__(self, session_maker: async_sessionmaker[AsyncSession]):
        """Bind the repository to an async session factory."""
        self.session_maker = session_maker

    async def create(self, session_id: str, user_id: int, name: str = "", username: str | None = None) -> ChatSession:  # noqa: E501
        """Create a new chat session."""
        async with self.session_maker() as session:
            chat_session = ChatSession(id=session_id, user_id=user_id, name=name, username=username)
            session.add(chat_session)
            await session.commit()
            await session.refresh(chat_session)
            logger.info("session_created", session_id=session_id, user_id=user_id, name=name)
            return chat_session

    async def delete(self, session_id: str) -> bool:
        """Delete a session by ID.

        Returns:
            bool: True if a session was deleted, False if none matched.
        """
        async with self.session_maker() as session:
            chat_session = await session.get(ChatSession, session_id)
            if not chat_session:
                return False
            await session.delete(chat_session)
            await session.commit()
            logger.info("session_deleted", session_id=session_id)
            return True

    async def get(self, session_id: str) -> ChatSession | None:
        """Get a session by ID."""
        async with self.session_maker() as session:
            return await session.get(ChatSession, session_id)

    async def list_for_user(self, user_id: int, limit: int = 20, offset: int = 0) -> list[ChatSession]:
        """List a page of a user's sessions, oldest first.

        Args:
            user_id: Owner of the sessions.
            limit: Max rows to return (caller is expected to bound this).
            offset: Rows to skip for pagination.
        """
        async with self.session_maker() as session:
            statement = (
                select(ChatSession)
                .where(col(ChatSession.user_id) == user_id)
                .order_by(col(ChatSession.created_at))
                .limit(limit)
                .offset(offset)
            )
            result = await session.exec(statement)
            return list(result.all())

    async def update_name(self, session_id: str, name: str) -> ChatSession:
        """Update a session's name.

        Raises:
            NotFoundError: If the session is not found.
        """
        async with self.session_maker() as session:
            chat_session = await session.get(ChatSession, session_id)
            if not chat_session:
                raise NotFoundError("Session", session_id)
            chat_session.name = name
            session.add(chat_session)
            await session.commit()
            await session.refresh(chat_session)
            logger.info("session_name_updated", session_id=session_id, name=name)
            return chat_session

    async def claim_name(self, session_id: str, placeholder: str) -> bool:
        """Atomically claim an unnamed session by writing a placeholder name.

        Executes ``UPDATE … WHERE name = ''`` in a single round-trip so exactly
        one concurrent caller wins (rowcount == 1).

        Returns:
            bool: True iff this caller won the claim.
        """
        async with self.session_maker() as session:
            stmt = (
                update(ChatSession)
                .where(col(ChatSession.id) == session_id, col(ChatSession.name) == "")
                .values(name=placeholder)
            )
            result = await session.exec(stmt)  # type: ignore[call-overload]
            await session.commit()
            return (result.rowcount or 0) == 1


# Singleton bound to the shared async session factory.
session_repository = SessionRepository(database.session_maker)
