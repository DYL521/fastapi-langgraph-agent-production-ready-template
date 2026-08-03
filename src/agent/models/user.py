"""This file contains the user model for the application."""

from typing import TYPE_CHECKING

import bcrypt
from sqlmodel import (
    Field,
    Relationship,
)

from agent.models.base import BaseModel

if TYPE_CHECKING:
    from agent.models.session import Session


class User(BaseModel, table=True):
    """User model for storing user accounts."""

    id: int | None = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    hashed_password: str
    username: str | None = Field(default=None, index=False)
    sessions: list["Session"] = Relationship(back_populates="user")

    def verify_password(self, password: str) -> bool:
        """Verify if the provided password matches the hash."""
        return bcrypt.checkpw(password.encode("utf-8"), self.hashed_password.encode("utf-8"))

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


# SQLModel/SQLAlchemy requires the related model class to be loaded before
# relationship resolution. This import ensures Session's mapper is registered.
from agent.models.session import Session  # noqa: E402, F811
