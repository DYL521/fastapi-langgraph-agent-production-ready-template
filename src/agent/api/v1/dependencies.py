"""Shared FastAPI dependencies: auth resolution and repository injection."""

from fastapi import (
    Depends,
    HTTPException,
    Request,
)
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from agent.core.logging import (
    bind_context,
    logger,
)
from agent.models.session import Session
from agent.models.user import User
from agent.repositories import (
    SessionRepository,
    UserRepository,
)
from agent.utils.auth import verify_token
from agent.utils.sanitization import sanitize_string

security = HTTPBearer()


def get_user_repository(request: Request) -> UserRepository:
    """Provide the shared user repository."""
    return UserRepository(request.app.state.database.session_maker)


def get_session_repository(request: Request) -> SessionRepository:
    """Provide the shared session repository."""
    return SessionRepository(request.app.state.database.session_maker)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    users: UserRepository = Depends(get_user_repository),
) -> User:
    """Resolve the authenticated user from the bearer token.

    Raises:
        HTTPException: If the token is invalid or the user does not exist.
    """
    try:
        token = sanitize_string(credentials.credentials)

        user_id = verify_token(token)
        if user_id is None:
            logger.error("invalid_token")
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user = await users.get(int(user_id))
        if user is None:
            logger.error("user_not_found", user_id=int(user_id))
            raise HTTPException(
                status_code=404,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        bind_context(user_id=user.id)
        return user
    except ValueError as ve:
        logger.exception("token_validation_failed", error=str(ve))
        raise HTTPException(
            status_code=422,
            detail="Invalid token format",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_session(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    sessions: SessionRepository = Depends(get_session_repository),
) -> Session:
    """Resolve the authenticated chat session from the bearer token.

    Raises:
        HTTPException: If the token is invalid or the session does not exist.
    """
    try:
        token = sanitize_string(credentials.credentials)

        session_id = verify_token(token)
        if session_id is None:
            logger.error("invalid_session_token")
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        session_id = sanitize_string(session_id)
        session = await sessions.get(session_id)
        if session is None:
            logger.error("session_not_found", session_id=session_id)
            raise HTTPException(
                status_code=404,
                detail="Session not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        bind_context(user_id=session.user_id)
        return session
    except ValueError as ve:
        logger.exception("token_validation_failed", error=str(ve))
        raise HTTPException(
            status_code=422,
            detail="Invalid token format",
            headers={"WWW-Authenticate": "Bearer"},
        )
