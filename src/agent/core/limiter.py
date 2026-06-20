"""Rate limiting configuration for the application.

This module configures rate limiting using slowapi, with default limits
defined in the application settings. Rate limits are applied based on
remote IP addresses.

When Valkey is configured, uses it as a distributed storage backend
so rate limits work correctly across multiple app instances.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from agent.core.config import settings
from agent.core.logging import logger

# Build storage URI for Valkey if configured
_storage_uri = None
if settings.cache.valkey_host:
    _password_part = f":{settings.cache.valkey_password}@" if settings.cache.valkey_password else ""
    _storage_uri = f"redis://{_password_part}{settings.cache.valkey_host}:{settings.cache.valkey_port}/{settings.cache.valkey_db}"
    logger.info("rate_limiter_using_valkey", host=settings.cache.valkey_host, port=settings.cache.valkey_port)

# Initialize rate limiter (uses in-memory storage if no Valkey)
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=settings.rate_limit.default,  # pyright: ignore[reportArgumentType]
    storage_uri=_storage_uri,
)
