"""LLM, long-term memory, and evaluation settings."""

from typing import Annotated

from pydantic import (
    Field,
    SecretStr,
    field_validator,
)
from pydantic_settings import (
    BaseSettings,
    NoDecode,
)

from agent.core.config._base import (
    settings_config,
    split_csv,
)


class LLMSettings(BaseSettings):
    """Primary LLM configuration.

    ``provider`` selects the chat-model backend (``openai`` and OpenAI-compatible
    endpoints by default; ``anthropic`` / ``bedrock`` / ``google`` when their
    optional extras are installed). Provider-specific credentials live in their
    own fields below and are only required when that provider is active.
    """

    model_config = settings_config()

    provider: str = Field(default="openai", validation_alias="LLM_PROVIDER")
    model: str = Field(default="gpt-5-mini", validation_alias="DEFAULT_LLM_MODEL")
    # Circular-fallback model list; empty means "no fallback, use `model` only".
    fallback_models: Annotated[list[str], NoDecode] = Field(
        default_factory=list, validation_alias="LLM_FALLBACK_MODELS"
    )
    temperature: float = Field(default=0.2, validation_alias="DEFAULT_LLM_TEMPERATURE")
    max_tokens: int = Field(default=2000, validation_alias="MAX_TOKENS")
    max_retries: int = Field(default=3, validation_alias="MAX_LLM_CALL_RETRIES")
    total_timeout: int = Field(default=60, validation_alias="LLM_TOTAL_TIMEOUT")
    session_naming_enabled: bool = Field(default=True, validation_alias="SESSION_NAMING_ENABLED")
    # Model used to auto-name sessions; empty falls back to the primary model.
    session_naming_model: str = Field(default="", validation_alias="SESSION_NAMING_MODEL")

    # --- OpenAI / OpenAI-compatible endpoints ---
    api_key: SecretStr = Field(default=SecretStr(""), validation_alias="OPENAI_API_KEY")
    base_url: str = Field(default="", validation_alias="OPENAI_BASE_URL")

    # --- Anthropic (native) ---
    anthropic_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="ANTHROPIC_API_KEY")

    @field_validator("provider", mode="before")
    @classmethod
    def _lower(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value

    @field_validator("fallback_models", mode="before")
    @classmethod
    def _split_fallback(cls, value: object) -> object:
        return split_csv(value)

    @property
    def model_chain(self) -> list[str]:
        """Ordered, de-duplicated model list: primary model then fallbacks."""
        chain = [self.model, *self.fallback_models]
        seen: set[str] = set()
        return [m for m in chain if m and not (m in seen or seen.add(m))]


class MemorySettings(BaseSettings):
    """Long-term memory (mem0) model and collection configuration.

    The LLM and embedder providers are independent of the agent's ``LLM_PROVIDER``
    (mem0 supports openai/anthropic/ollama/huggingface/… directly).
    """

    model_config = settings_config(env_prefix="LONG_TERM_MEMORY_")

    llm_provider: str = Field(default="openai")
    model: str = Field(default="gpt-5-nano")
    embedder_provider: str = Field(default="openai")
    embedder_model: str = Field(default="text-embedding-3-small")
    collection_name: str = Field(default="longterm_memory")


class EvaluationSettings(BaseSettings):
    """LLM evaluation settings; ``api_key`` falls back to the LLM key at root."""

    model_config = settings_config(env_prefix="EVALUATION_")

    llm: str = Field(default="gpt-5")
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: SecretStr = Field(default=SecretStr(""))
    sleep_time: int = Field(default=10)
