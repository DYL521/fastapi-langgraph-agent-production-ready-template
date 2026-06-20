"""Chat session persistence (CRUD) backed by SQLModel."""

from typing import (
    List,
    Optional,
)

from fastapi import HTTPException
from sqlalchemy.engine import Engine
from sqlmodel import (
    Session,
    col,
    select,
    update,
)

from agent.core.logging import logger
from agent.models.session import Session as ChatSession
from agent.services.database import database


class SessionRepository:
    """CRUD operations for chat ``Session`` records."""

    def __init__(self, engine: Engine):
        """Bind the repository to a SQLAlchemy engine."""
        self.engine = engine

    async def create(self, session_id: str, user_id: int, name: str = "", username: str | None = None) -> ChatSession:
        """Create a new chat session."""
        with Session(self.engine) as session:
            chat_session = ChatSession(id=session_id, user_id=user_id, name=name, username=username)
            session.add(chat_session)
            session.commit()
            session.refresh(chat_session)
            logger.info("session_created", session_id=session_id, user_id=user_id, name=name)
            return chat_session

    async def delete(self, session_id: str) -> bool:
        """Delete a session by ID.

        Returns:
            bool: True if a session was deleted, False if none matched.
        """
        with Session(self.engine) as session:
            chat_session = session.get(ChatSession, session_id)
            if not chat_session:
                return False
            session.delete(chat_session)
            session.commit()
            logger.info("session_deleted", session_id=session_id)
            return True

    async def get(self, session_id: str) -> Optional[ChatSession]:
        """Get a session by ID."""
        with Session(self.engine) as session:
            return session.get(ChatSession, session_id)

    async def list_for_user(self, user_id: int) -> List[ChatSession]:
        """List all sessions for a user, oldest first."""
        with Session(self.engine) as session:
            statement = (
                select(ChatSession).where(col(ChatSession.user_id) == user_id).order_by(col(ChatSession.created_at))
            )
            return list(session.exec(statement).all())

    async def update_name(self, session_id: str, name: str) -> ChatSession:
        """Update a session's name.

        Raises:
            HTTPException: If the session is not found.
        """
        with Session(self.engine) as session:
            chat_session = session.get(ChatSession, session_id)
            if not chat_session:
                raise HTTPException(status_code=404, detail="Session not found")
            chat_session.name = name
            session.add(chat_session)
            session.commit()
            session.refresh(chat_session)
            logger.info("session_name_updated", session_id=session_id, name=name)
            return chat_session

    def claim_name(self, session_id: str, placeholder: str) -> bool:
        """Atomically claim an unnamed session by writing a placeholder name.

        Executes ``UPDATE … WHERE name = ''`` in a single round-trip so exactly
        one concurrent caller wins (rowcount == 1). Synchronous on purpose:
        callers invoke it from sync session-naming code.

        Returns:
            bool: True iff this caller won the claim.
        """
        with Session(self.engine) as session:
            stmt = (
                update(ChatSession)
                .where(col(ChatSession.id) == session_id, col(ChatSession.name) == "")
                .values(name=placeholder)
            )
            result = session.exec(stmt)  # type: ignore[call-overload]
            session.commit()
            return (result.rowcount or 0) == 1


# Singleton bound to the shared engine.
session_repository = SessionRepository(database.engine)
