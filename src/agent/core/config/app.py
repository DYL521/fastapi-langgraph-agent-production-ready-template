"""Core application settings."""

from typing import Annotated

from pydantic import (
    Field,
    field_validator,
)
from pydantic_settings import (
    BaseSettings,
    NoDecode,
)

from agent.core.config._base import (
    Environment,
    get_environment,
    settings_config,
    split_csv,
)


class AppSettings(BaseSettings):
    """Project metadata, API prefix, debug flag, CORS, and environment."""

    model_config = settings_config()

    project_name: str = Field(default="FastAPI LangGraph Template", validation_alias="PROJECT_NAME")
    version: str = Field(default="1.0.0", validation_alias="VERSION")
    description: str = Field(
        default="A production-ready FastAPI template with LangGraph and Langfuse integration",
        validation_alias="DESCRIPTION",
    )
    api_v1_str: str = Field(default="/api/v1", validation_alias="API_V1_STR")
    debug: bool = Field(default=False, validation_alias="DEBUG")
    allowed_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["*"], validation_alias="ALLOWED_ORIGINS"
    )
    environment: Environment = Field(default_factory=get_environment)

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        return split_csv(value)
