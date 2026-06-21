"""Application configuration (pydantic-settings, split by domain).

Access is nested: ``settings.database.host``, ``settings.llm.model``,
``settings.jwt.secret_key``, etc. The import path ``agent.core.config`` and the
``settings`` / ``Environment`` exports are preserved for callers.
"""

from pydantic import (
    BaseModel,
    model_validator,
)

from agent.core.config._base import (
    Environment,
    get_environment,
)
from agent.core.config.app import AppSettings
from agent.core.config.database import (
    DatabaseSettings,
    VectorStoreSettings,
)
from agent.core.config.infra import (
    CacheSettings,
    RateLimitSettings,
)
from agent.core.config.llm import (
    EvaluationSettings,
    LLMSettings,
    MemorySettings,
)
from agent.core.config.observability import (
    LangfuseSettings,
    LoggingSettings,
)
from agent.core.config.security import JWTSettings

# Environment-specific defaults, applied only to fields not explicitly set.
_ENV_DEFAULTS: dict[Environment, dict[str, object]] = {
    Environment.DEVELOPMENT: {
        "debug": True,
        "log_level": "DEBUG",
        "log_format": "console",
        "rate_limit_default": ["1000 per day", "200 per hour"],
    },
    Environment.STAGING: {
        "debug": False,
        "log_level": "INFO",
        "rate_limit_default": ["500 per day", "100 per hour"],
    },
    Environment.PRODUCTION: {
        "debug": False,
        "log_level": "WARNING",
        "rate_limit_default": ["200 per day", "50 per hour"],
    },
    Environment.TEST: {
        "debug": True,
        "log_level": "DEBUG",
        "log_format": "console",
        "rate_limit_default": ["1000 per day", "1000 per hour"],
    },
}


class Settings(BaseModel):
    """Root settings composing all domain sub-settings."""

    app: AppSettings
    database: DatabaseSettings
    vector: VectorStoreSettings
    llm: LLMSettings
    memory: MemorySettings
    evaluation: EvaluationSettings
    langfuse: LangfuseSettings
    logging: LoggingSettings
    jwt: JWTSettings
    cache: CacheSettings
    rate_limit: RateLimitSettings

    @model_validator(mode="after")
    def _resolve_cross_cutting(self) -> "Settings":
        # Vector store provider defaults to weaviate on mysql, else pgvector.
        if "provider" not in self.vector.model_fields_set or not self.vector.provider:
            self.vector.provider = "weaviate" if self.database.dialect == "mysql" else "pgvector"

        # Evaluation API key falls back to the main LLM key when unset.
        if "api_key" not in self.evaluation.model_fields_set or not self.evaluation.api_key.get_secret_value():
            self.evaluation.api_key = self.llm.api_key

        # Apply environment-specific defaults only where not explicitly set.
        env_defaults = _ENV_DEFAULTS.get(self.app.environment, {})
        if "debug" in env_defaults and "debug" not in self.app.model_fields_set:
            self.app.debug = bool(env_defaults["debug"])
        if "log_level" in env_defaults and "level" not in self.logging.model_fields_set:
            self.logging.level = str(env_defaults["log_level"])
        if "log_format" in env_defaults and "format" not in self.logging.model_fields_set:
            self.logging.format = str(env_defaults["log_format"])
        if "rate_limit_default" in env_defaults and "default" not in self.rate_limit.model_fields_set:
            self.rate_limit.default = list(env_defaults["rate_limit_default"])  # type: ignore[arg-type]
        return self


def _build_settings() -> Settings:
    """Instantiate each domain settings class (each reads env + .env files)."""
    return Settings(
        app=AppSettings(),
        database=DatabaseSettings(),
        vector=VectorStoreSettings(),
        llm=LLMSettings(),
        memory=MemorySettings(),
        evaluation=EvaluationSettings(),
        langfuse=LangfuseSettings(),
        logging=LoggingSettings(),
        jwt=JWTSettings(),
        cache=CacheSettings(),
        rate_limit=RateLimitSettings(),
    )


settings = _build_settings()

__all__ = ["settings", "Settings", "Environment", "get_environment"]
