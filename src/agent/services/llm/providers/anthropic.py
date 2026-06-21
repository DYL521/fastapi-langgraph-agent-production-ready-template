"""Anthropic (native) provider adapter.

Only imported when ``LLM_PROVIDER=anthropic`` and the ``anthropic`` extra is
installed; the third-party imports are ``# type: ignore``'d so a minimal
(OpenAI-only) install still passes type checks.
"""

from typing import (
    Any,
    override,
)

from anthropic import (  # type: ignore[import-not-found]
    AnthropicError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OverloadedError,
    RateLimitError,
)
from langchain_anthropic import ChatAnthropic  # type: ignore[import-not-found]
from langchain_core.language_models.chat_models import BaseChatModel

from agent.core.config import settings
from agent.services.llm.providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Builds ``ChatAnthropic`` instances and classifies Anthropic SDK errors."""

    retryable_errors = (RateLimitError, APITimeoutError, APIConnectionError, OverloadedError, InternalServerError)
    fatal_errors = (AnthropicError,)

    @override
    def build(self, model_name: str, **overrides: Any) -> BaseChatModel:
        """Construct a ``ChatAnthropic`` for ``model_name``.

        Anthropic requires ``max_tokens``; it defaults from settings unless the
        caller overrides it.
        """
        params: dict[str, Any] = {
            "model": model_name,
            "anthropic_api_key": settings.llm.anthropic_api_key,
            "max_tokens": settings.llm.max_tokens,
        }
        params.update(overrides)
        return ChatAnthropic(**params)
