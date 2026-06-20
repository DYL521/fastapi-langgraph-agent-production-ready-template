"""User persistence (CRUD) backed by SQLModel."""

from typing import Optional

from sqlalchemy.engine import Engine
from sqlmodel import (
    Session,
    select,
)

from agent.core.logging import logger
from agent.models.user import User
from agent.services.database import database


class UserRepository:
    """CRUD operations for ``User`` records."""

    def __init__(self, engine: Engine):
        """Bind the repository to a SQLAlchemy engine."""
        self.engine = engine

    async def create(self, email: str, password: str, username: str | None = None) -> User:
        """Create a new user.

        Args:
            email: User's email address.
            password: Hashed password.
            username: Optional display name.

        Returns:
            User: The created user.
        """
        with Session(self.engine) as session:
            user = User(email=email, hashed_password=password, username=username)
            session.add(user)
            session.commit()
            session.refresh(user)
            logger.info("user_created", email=email)
            return user

    async def get(self, user_id: int) -> Optional[User]:
        """Get a user by ID."""
        with Session(self.engine) as session:
            return session.get(User, user_id)

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        with Session(self.engine) as session:
            return session.exec(select(User).where(User.email == email)).first()

    async def delete_by_email(self, email: str) -> bool:
        """Delete a user by email.

        Returns:
            bool: True if a user was deleted, False if none matched.
        """
        with Session(self.engine) as session:
            user = session.exec(select(User).where(User.email == email)).first()
            if not user:
                return False
            session.delete(user)
            session.commit()
            logger.info("user_deleted", email=email)
            return True


# Singleton bound to the shared engine.
user_repository = UserRepository(database.engine)
