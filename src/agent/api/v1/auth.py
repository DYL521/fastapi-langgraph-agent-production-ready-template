"""Authentication and authorization endpoints for the API.

This module provides endpoints for user registration, login, session management,
and token verification. Auth resolution and repository injection live in
``agent.api.v1.dependencies``.
"""

import uuid
from typing import List

from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    Request,
)

from agent.api.v1.dependencies import (
    get_current_session,
    get_current_user,
    get_session_repository,
    get_user_repository,
)
from agent.core.config import settings
from agent.core.limiter import limiter
from agent.core.logging import logger
from agent.models.session import Session
from agent.models.user import User
from agent.repositories import (
    SessionRepository,
    UserRepository,
)
from agent.schemas.auth import (
    SessionResponse,
    TokenResponse,
    UserCreate,
    UserResponse,
)
from agent.utils.auth import create_access_token
from agent.utils.sanitization import (
    sanitize_email,
    sanitize_string,
    validate_password_strength,
)

router = APIRouter()


@router.post("/register", response_model=UserResponse)
@limiter.limit(settings.rate_limit.endpoints["register"][0])
async def register_user(
    request: Request,
    user_data: UserCreate,
    users: UserRepository = Depends(get_user_repository),
):
    """Register a new user.

    Args:
        request: The FastAPI request object for rate limiting.
        user_data: User registration data.
        users: Injected user repository.

    Returns:
        UserResponse: The created user info.
    """
    try:
        sanitized_email = sanitize_email(user_data.email)

        password = user_data.password.get_secret_value()
        validate_password_strength(password)

        if await users.get_by_email(sanitized_email):
            raise HTTPException(status_code=400, detail="Email already registered")

        sanitized_username = sanitize_string(user_data.username) if user_data.username else None

        user = await users.create(
            email=sanitized_email,
            password=User.hash_password(password),
            username=sanitized_username,
        )

        token = create_access_token(str(user.id))
        return UserResponse(id=user.id, email=user.email, username=user.username, token=token)
    except ValueError as ve:
        logger.exception("user_registration_validation_failed", error=str(ve))
        raise HTTPException(status_code=422, detail=str(ve))


@router.post("/login", response_model=TokenResponse)
@limiter.limit(settings.rate_limit.endpoints["login"][0])
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    grant_type: str = Form(default="password"),
    users: UserRepository = Depends(get_user_repository),
):
    """Login a user.

    Args:
        request: The FastAPI request object for rate limiting.
        email: User's email.
        password: User's password.
        grant_type: Must be "password".
        users: Injected user repository.

    Returns:
        TokenResponse: Access token information.

    Raises:
        HTTPException: If credentials are invalid.
    """
    try:
        email = sanitize_string(email)
        password = sanitize_string(password)
        grant_type = sanitize_string(grant_type)

        if grant_type != "password":
            raise HTTPException(
                status_code=400,
                detail="Unsupported grant type. Must be 'password'",
            )

        user = await users.get_by_email(email)
        if not user or not user.verify_password(password):
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = create_access_token(str(user.id))
        return TokenResponse(access_token=token.access_token, token_type="bearer", expires_at=token.expires_at)
    except ValueError as ve:
        logger.exception("login_validation_failed", error=str(ve))
        raise HTTPException(status_code=422, detail=str(ve))


@router.post("/session", response_model=SessionResponse)
async def create_session(
    user: User = Depends(get_current_user),
    sessions: SessionRepository = Depends(get_session_repository),
):
    """Create a new chat session for the authenticated user.

    Args:
        user: The authenticated user.
        sessions: Injected session repository.

    Returns:
        SessionResponse: The session ID, name, and access token.
    """
    try:
        session_id = str(uuid.uuid4())
        session = await sessions.create(session_id, user.id, username=user.username)
        token = create_access_token(session_id)

        logger.info(
            "session_created",
            session_id=session_id,
            user_id=user.id,
            name=session.name,
            expires_at=token.expires_at.isoformat(),
        )

        return SessionResponse(session_id=session_id, name=session.name, token=token)
    except ValueError as ve:
        logger.exception("session_creation_validation_failed", error=str(ve), user_id=user.id)
        raise HTTPException(status_code=422, detail=str(ve))


@router.patch("/session/{session_id}/name", response_model=SessionResponse)
async def update_session_name(
    session_id: str,
    name: str = Form(...),
    current_session: Session = Depends(get_current_session),
    sessions: SessionRepository = Depends(get_session_repository),
):
    """Update a session's name.

    Args:
        session_id: The ID of the session to update.
        name: The new name for the session.
        current_session: The current session from auth.
        sessions: Injected session repository.

    Returns:
        SessionResponse: The updated session information.
    """
    try:
        sanitized_session_id = sanitize_string(session_id)
        sanitized_name = sanitize_string(name)
        sanitized_current_session = sanitize_string(current_session.id)

        if sanitized_session_id != sanitized_current_session:
            raise HTTPException(status_code=403, detail="Cannot modify other sessions")

        session = await sessions.update_name(sanitized_session_id, sanitized_name)
        token = create_access_token(sanitized_session_id)

        return SessionResponse(session_id=sanitized_session_id, name=session.name, token=token)
    except ValueError as ve:
        logger.exception("session_update_validation_failed", error=str(ve), session_id=session_id)
        raise HTTPException(status_code=422, detail=str(ve))


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    current_session: Session = Depends(get_current_session),
    sessions: SessionRepository = Depends(get_session_repository),
):
    """Delete a session for the authenticated user.

    Args:
        session_id: The ID of the session to delete.
        current_session: The current session from auth.
        sessions: Injected session repository.
    """
    try:
        sanitized_session_id = sanitize_string(session_id)
        sanitized_current_session = sanitize_string(current_session.id)

        if sanitized_session_id != sanitized_current_session:
            raise HTTPException(status_code=403, detail="Cannot delete other sessions")

        await sessions.delete(sanitized_session_id)
        logger.info("session_deleted", session_id=session_id, user_id=current_session.user_id)
    except ValueError as ve:
        logger.exception("session_deletion_validation_failed", error=str(ve), session_id=session_id)
        raise HTTPException(status_code=422, detail=str(ve))


@router.get("/sessions", response_model=List[SessionResponse])
async def get_user_sessions(
    user: User = Depends(get_current_user),
    sessions: SessionRepository = Depends(get_session_repository),
):
    """Get all sessions for the authenticated user.

    Args:
        user: The authenticated user.
        sessions: Injected session repository.

    Returns:
        List[SessionResponse]: List of the user's sessions.
    """
    try:
        user_sessions = await sessions.list_for_user(user.id)
        return [
            SessionResponse(
                session_id=sanitize_string(session.id),
                name=sanitize_string(session.name),
                token=create_access_token(session.id),
            )
            for session in user_sessions
        ]
    except ValueError as ve:
        logger.exception("get_sessions_validation_failed", user_id=user.id, error=str(ve))
        raise HTTPException(status_code=422, detail=str(ve))
