"""User persistence (CRUD) backed by async SQLModel."""

from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from agent.core.logging import logger
from agent.models.user import User
from agent.services.database import database


class UserRepository:
    """CRUD operations for ``User`` records."""

    def __init__(self, session_maker: async_sessionmaker[AsyncSession]):
        """Bind the repository to an async session factory."""
        self.session_maker = session_maker

    async def create(self, email: str, password: str, username: str | None = None) -> User:
        """Create a new user.

        Args:
            email: User's email address.
            password: Hashed password.
            username: Optional display name.

        Returns:
            User: The created user.
        """
        async with self.session_maker() as session:
            user = User(email=email, hashed_password=password, username=username)
            session.add(user)
            await session.commit()
            await session.refresh(user)
            logger.info("user_created", email=email)
            return user

    async def get(self, user_id: int) -> User | None:
        """Get a user by ID."""
        async with self.session_maker() as session:
            return await session.get(User, user_id)

    async def get_by_email(self, email: str) -> User | None:
        """Get a user by email."""
        async with self.session_maker() as session:
            result = await session.exec(select(User).where(User.email == email))
            return result.first()

    async def delete_by_email(self, email: str) -> bool:
        """Delete a user by email.

        Returns:
            bool: True if a user was deleted, False if none matched.
        """
        async with self.session_maker() as session:
            result = await session.exec(select(User).where(User.email == email))
            user = result.first()
            if not user:
                return False
            await session.delete(user)
            await session.commit()
            logger.info("user_deleted", email=email)
            return True


# Singleton bound to the shared async session factory.
user_repository = UserRepository(database.session_maker)
