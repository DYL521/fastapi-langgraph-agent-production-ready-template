"""Langfuse tracing and logging/profiling settings."""

from pathlib import Path

from pydantic import (
    Field,
    SecretStr,
)
from pydantic_settings import BaseSettings

from agent.core.config._base import settings_config


class LangfuseSettings(BaseSettings):
    """Langfuse observability configuration."""

    model_config = settings_config(env_prefix="LANGFUSE_")

    tracing_enabled: bool = Field(default=True)
    public_key: str = Field(default="")
    secret_key: SecretStr = Field(default=SecretStr(""))
    host: str = Field(default="https://cloud.langfuse.com")


class LoggingSettings(BaseSettings):
    """Structured logging and DEBUG-only profiling configuration."""

    model_config = settings_config()

    dir: Path = Field(default=Path("logs"), validation_alias="LOG_DIR")
    level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    format: str = Field(default="json", validation_alias="LOG_FORMAT")
    profiling_dir: Path = Field(default=Path("/tmp/fastapi_profiles"), validation_alias="PROFILING_DIR")
    profiling_threshold_seconds: float = Field(default=2.0, validation_alias="PROFILING_THRESHOLD_SECONDS")
