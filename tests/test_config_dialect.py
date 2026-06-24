"""Unit tests for dialect-aware settings resolution and POSTGRES_* compatibility.

These build settings with the dotenv files disabled (``_env_file=None``) so the
repo's own ``.env`` cannot leak values and skew default-derivation assertions.
"""

from agent.core import config as config_module
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

_CLEARED_KEYS = (
    "DB_DIALECT",
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "VECTOR_STORE_PROVIDER",
    "EVALUATION_API_KEY",
    "OPENAI_API_KEY",
)


def _isolated(monkeypatch, **env):
    """Build a full Settings from os.environ only (dotenv disabled)."""
    for key in _CLEARED_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return config_module.Settings(
        app=AppSettings(_env_file=None),
        database=DatabaseSettings(_env_file=None),
        vector=VectorStoreSettings(_env_file=None),
        llm=LLMSettings(_env_file=None),
        memory=MemorySettings(_env_file=None),
        evaluation=EvaluationSettings(_env_file=None),
        langfuse=LangfuseSettings(_env_file=None),
        logging=LoggingSettings(_env_file=None),
        jwt=JWTSettings(_env_file=None),
        cache=CacheSettings(_env_file=None),
        rate_limit=RateLimitSettings(_env_file=None),
    )


def test_postgres_alias_backward_compat(monkeypatch):
    s = _isolated(monkeypatch, POSTGRES_HOST="legacy-host", POSTGRES_DB="legacy-db")
    assert s.database.host == "legacy-host"
    assert s.database.name == "legacy-db"


def test_db_takes_precedence_over_postgres(monkeypatch):
    s = _isolated(monkeypatch, DB_HOST="new-host", POSTGRES_HOST="old-host")
    assert s.database.host == "new-host"


def test_postgres_dialect_defaults(monkeypatch):
    s = _isolated(monkeypatch, DB_DIALECT="postgres")
    assert s.database.port == 5432
    assert s.vector.provider == "pgvector"


def test_mysql_dialect_defaults(monkeypatch):
    s = _isolated(monkeypatch, DB_DIALECT="mysql")
    assert s.database.port == 3306
    assert s.vector.provider == "weaviate"


def test_explicit_vector_provider_overrides_dialect_default(monkeypatch):
    s = _isolated(monkeypatch, DB_DIALECT="mysql", VECTOR_STORE_PROVIDER="pgvector")
    assert s.vector.provider == "pgvector"


def test_evaluation_api_key_falls_back_to_llm_key(monkeypatch):
    s = _isolated(monkeypatch, OPENAI_API_KEY="sk-from-llm")
    assert s.evaluation.api_key.get_secret_value() == "sk-from-llm"
