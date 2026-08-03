"""Observability module for the application."""

from langfuse import Langfuse

from agent.core.config import settings
from agent.core.logging import logger


def langfuse_init() -> None:
    """Initialize Langfuse tracing (call once at startup)."""
    if not settings.langfuse.tracing_enabled:
        logger.debug("langfuse_tracing_disabled")
        return

    langfuse = Langfuse(
        tracing_enabled=settings.langfuse.tracing_enabled,
        public_key=settings.langfuse.public_key,
        secret_key=settings.langfuse.secret_key.get_secret_value(),
        host=settings.langfuse.host,
        environment=settings.app.environment.value,
        debug=settings.app.debug,
    )

    try:
        if langfuse.auth_check():
            logger.debug("langfuse_auth_success")
        else:
            logger.warning("langfuse_auth_failure")
    except Exception:
        logger.exception("langfuse_auth_check_failed")
