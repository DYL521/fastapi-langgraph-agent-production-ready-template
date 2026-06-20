"""LLM, long-term memory, and evaluation settings."""

from pydantic import Field
from pydantic_settings import BaseSettings

from agent.core.config._base import settings_config


class LLMSettings(BaseSettings):
    """Primary LLM configuration."""

    model_config = settings_config()

    api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    model: str = Field(default="gpt-5-mini", validation_alias="DEFAULT_LLM_MODEL")
    temperature: float = Field(default=0.2, validation_alias="DEFAULT_LLM_TEMPERATURE")
    max_tokens: int = Field(default=2000, validation_alias="MAX_TOKENS")
    max_retries: int = Field(default=3, validation_alias="MAX_LLM_CALL_RETRIES")
    total_timeout: int = Field(default=60, validation_alias="LLM_TOTAL_TIMEOUT")
    session_naming_enabled: bool = Field(default=True, validation_alias="SESSION_NAMING_ENABLED")


class MemorySettings(BaseSettings):
    """Long-term memory (mem0) model and collection configuration."""

    model_config = settings_config(env_prefix="LONG_TERM_MEMORY_")

    model: str = Field(default="gpt-5-nano")
    embedder_model: str = Field(default="text-embedding-3-small")
    collection_name: str = Field(default="longterm_memory")


class EvaluationSettings(BaseSettings):
    """LLM evaluation settings; ``api_key`` falls back to the LLM key at root."""

    model_config = settings_config(env_prefix="EVALUATION_")

    llm: str = Field(default="gpt-5")
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: str = Field(default="")
    sleep_time: int = Field(default=10)
