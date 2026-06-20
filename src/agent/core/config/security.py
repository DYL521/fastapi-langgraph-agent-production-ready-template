"""Authentication / JWT settings."""

from pydantic import Field
from pydantic_settings import BaseSettings

from agent.core.config._base import settings_config


class JWTSettings(BaseSettings):
    """JWT signing and expiry configuration."""

    model_config = settings_config(env_prefix="JWT_")

    secret_key: str = Field(default="")
    algorithm: str = Field(default="HS256")
    access_token_expire_days: int = Field(default=30)
