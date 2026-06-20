"""Relational database and vector store settings.

Generalized ``DB_*`` keys with ``POSTGRES_*`` backward-compatible fallbacks, so
existing PostgreSQL deployments keep working without changes.
"""

from typing import Optional

from pydantic import (
    AliasChoices,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings

from agent.core.config._base import settings_config


class DatabaseSettings(BaseSettings):
    """Dialect-aware relational settings (``postgres`` | ``mysql``)."""

    model_config = settings_config()

    dialect: str = Field(default="postgres", validation_alias="DB_DIALECT")
    host: str = Field(default="localhost", validation_alias=AliasChoices("DB_HOST", "POSTGRES_HOST"))
    port: Optional[int] = Field(default=None, validation_alias=AliasChoices("DB_PORT", "POSTGRES_PORT"))
    name: str = Field(default="food_order_db", validation_alias=AliasChoices("DB_NAME", "POSTGRES_DB"))
    user: str = Field(default="postgres", validation_alias=AliasChoices("DB_USER", "POSTGRES_USER"))
    password: str = Field(default="postgres", validation_alias=AliasChoices("DB_PASSWORD", "POSTGRES_PASSWORD"))
    pool_size: int = Field(default=20, validation_alias=AliasChoices("DB_POOL_SIZE", "POSTGRES_POOL_SIZE"))
    max_overflow: int = Field(default=10, validation_alias=AliasChoices("DB_MAX_OVERFLOW", "POSTGRES_MAX_OVERFLOW"))
    checkpoint_tables: list[str] = Field(
        default_factory=lambda: ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]
    )

    @field_validator("dialect", mode="before")
    @classmethod
    def _lower(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _default_port(self) -> "DatabaseSettings":
        if self.port is None:
            self.port = 3306 if self.dialect == "mysql" else 5432
        return self


class VectorStoreSettings(BaseSettings):
    """Long-term memory vector store (``pgvector`` | ``weaviate``).

    ``provider`` defaults are resolved from the DB dialect at the root level.
    """

    model_config = settings_config()

    provider: str = Field(default="", validation_alias="VECTOR_STORE_PROVIDER")
    weaviate_cluster_url: str = Field(default="http://localhost:8080", validation_alias="WEAVIATE_CLUSTER_URL")
    weaviate_api_key: str = Field(default="", validation_alias="WEAVIATE_API_KEY")

    @field_validator("provider", mode="before")
    @classmethod
    def _lower(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value
