"""This file contains the session model for the application."""

from typing import (
    TYPE_CHECKING,
    Optional,
)

from sqlalchemy import Index
from sqlmodel import (
    Field,
    Relationship,
)

from agent.models.base import BaseModel

if TYPE_CHECKING:
    from agent.models.user import User


class Session(BaseModel, table=True):
    """Session model for storing chat sessions.

    Attributes:
        id: The primary key
        user_id: Foreign key to the user
        name: Name of the session (defaults to empty string)
        username: Display name copied from the user at session creation
        created_at: When the session was created
        messages: Relationship to session messages
        user: Relationship to the session owner
    """

    # Composite index serves `WHERE user_id = ? ORDER BY created_at` (list_for_user)
    # and the user_id-prefix lookup, avoiding a sequential scan + sort.
    __table_args__ = (Index("ix_session_user_created", "user_id", "created_at"),)

    id: str = Field(primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    name: str = Field(default="")
    username: Optional[str] = Field(default=None)
    user: "User" = Relationship(back_populates="sessions")
