"""Cache (Valkey/Redis) and rate-limiting settings."""

import os
from typing import Annotated

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    NoDecode,
)

from agent.core.config._base import (
    settings_config,
    split_csv,
)


class CacheSettings(BaseSettings):
    """Optional Valkey/Redis cache; caching is enabled when ``valkey_host`` is set."""

    model_config = settings_config()

    valkey_host: str = Field(default="", validation_alias="VALKEY_HOST")
    valkey_port: int = Field(default=6379, validation_alias="VALKEY_PORT")
    valkey_db: int = Field(default=0, validation_alias="VALKEY_DB")
    valkey_password: SecretStr = Field(default=SecretStr(""), validation_alias="VALKEY_PASSWORD")
    valkey_max_connections: int = Field(default=20, validation_alias="VALKEY_MAX_CONNECTIONS")
    ttl_seconds: int = Field(default=60, validation_alias="CACHE_TTL_SECONDS")
    max_items: int = Field(default=10000, validation_alias="CACHE_MAX_ITEMS")


def _default_endpoints() -> dict[str, list[str]]:
    return {
        "chat": ["30 per minute"],
        "chat_stream": ["20 per minute"],
        "messages": ["50 per minute"],
        "register": ["10 per hour"],
        "login": ["20 per minute"],
        "root": ["10 per minute"],
        "health": ["20 per minute"],
    }


class RateLimitSettings(BaseSettings):
    """Default and per-endpoint rate limits.

    Per-endpoint overrides are read from ``RATE_LIMIT_<ENDPOINT>`` env vars.
    """

    model_config = settings_config()

    default: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["200 per day", "50 per hour"], validation_alias="RATE_LIMIT_DEFAULT"
    )
    endpoints: dict[str, list[str]] = Field(default_factory=_default_endpoints)

    @field_validator("default", mode="before")
    @classmethod
    def _split_default(cls, value: object) -> object:
        return split_csv(value)

    @model_validator(mode="after")
    def _load_endpoint_overrides(self) -> "RateLimitSettings":
        for endpoint in list(self.endpoints):
            raw = os.getenv(f"RATE_LIMIT_{endpoint.upper()}")
            if raw:
                parsed = split_csv(raw)
                if parsed:
                    self.endpoints[endpoint] = parsed  # type: ignore[assignment]
        return self
