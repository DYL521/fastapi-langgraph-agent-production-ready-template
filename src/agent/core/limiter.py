"""Rate limiting configuration for the application.

This module configures rate limiting using slowapi, with default limits
defined in the application settings. Rate limits are applied based on
remote IP addresses.

When Valkey is configured, uses it as a distributed storage backend
so rate limits work correctly across multiple app instances.

The ``limiter`` instance must be module-level because slowapi's
``@limiter.limit(...)`` decorator captures it at import time.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from agent.core.config import settings
from agent.core.logging import logger

_storage_uri = None
if settings.cache.valkey_host:
    _valkey_password = settings.cache.valkey_password.get_secret_value()
    _password_part = f":{_valkey_password}@" if _valkey_password else ""
    _storage_uri = f"redis://{_password_part}{settings.cache.valkey_host}:{settings.cache.valkey_port}/{settings.cache.valkey_db}"
    logger.info("rate_limiter_using_valkey", host=settings.cache.valkey_host, port=settings.cache.valkey_port)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=settings.rate_limit.default,  # pyright: ignore[reportArgumentType]
    storage_uri=_storage_uri,
)
