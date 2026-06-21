"""OpenAI (and OpenAI-compatible endpoint) provider adapter.

Covers the OpenAI API directly as well as any wire-compatible endpoint
(DeepSeek, Together, vLLM, Ollama, Atlas Cloud, …) selected via ``OPENAI_BASE_URL``.
"""

from functools import cached_property
from typing import (
    Any,
    override,
)

import tiktoken
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from openai import (
    APIError,
    APITimeoutError,
    OpenAIError,
    RateLimitError,
)

from agent.core.config import settings
from agent.services.llm.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """Builds ``ChatOpenAI`` instances and classifies OpenAI SDK errors."""

    retryable_errors = (RateLimitError, APITimeoutError, APIError)
    fatal_errors = (OpenAIError,)

    @override
    def build(self, model_name: str, **overrides: Any) -> BaseChatModel:
        """Construct a ``ChatOpenAI`` for ``model_name``.

        Applies the configured API key and (optional) base URL, defaulting the
        token cap via ``model_kwargs`` unless the caller overrides it.
        """
        params: dict[str, Any] = {
            "model": model_name,
            "api_key": settings.llm.api_key,
        }
        if settings.llm.base_url:
            params["base_url"] = settings.llm.base_url
        params.update(overrides)
        params.setdefault("model_kwargs", {"max_completion_tokens": settings.llm.max_tokens})
        return ChatOpenAI(**params)

    @cached_property
    def _encoding(self) -> tiktoken.Encoding:
        """Return the tiktoken encoding for the configured model, cached on the adapter."""
        try:
            return tiktoken.encoding_for_model(settings.llm.model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    @override
    def count_tokens(self, text: str) -> int:
        """Exact local token count via tiktoken."""
        return len(self._encoding.encode(text)) if text else 0
