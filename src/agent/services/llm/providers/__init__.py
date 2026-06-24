"""LLM provider factory.

Resolves the active provider adapter from ``settings.llm.provider`` and exposes
a single ``build_chat_model`` entry point. Non-default providers import their
optional SDKs lazily so a minimal install (OpenAI only) stays importable.
"""

from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from agent.core.config import settings
from agent.services.llm.providers.base import LLMProvider


@lru_cache(maxsize=None)
def get_active_provider() -> LLMProvider:
    """Return the adapter for the configured ``settings.llm.provider``.

    Raises:
        ValueError: If the configured provider is not supported.
    """
    provider = settings.llm.provider

    if provider == "openai":
        from agent.services.llm.providers.openai import OpenAIProvider

        return OpenAIProvider()

    if provider == "anthropic":
        from agent.services.llm.providers.anthropic import AnthropicProvider

        return AnthropicProvider()

    raise ValueError(f"unsupported LLM_PROVIDER: {provider!r} (expected 'openai' or 'anthropic')")


def build_chat_model(model_name: str, **overrides: Any) -> BaseChatModel:
    """Build a chat model for ``model_name`` using the active provider."""
    return get_active_provider().build(model_name, **overrides)


__all__ = ["LLMProvider", "build_chat_model", "get_active_provider"]
