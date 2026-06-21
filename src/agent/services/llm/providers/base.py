"""Provider adapter contract for chat-model construction and error classification.

Each provider adapter knows how to build a LangChain ``BaseChatModel`` for the
active backend and declares which exceptions are transient (retry the same
model) versus fatal (fall back to the next model). This keeps ``LLMService``
free of any provider-specific exception types.
"""

from abc import (
    ABC,
    abstractmethod,
)
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel


class LLMProvider(ABC):
    """Base class for chat-model provider adapters."""

    #: Exceptions that should trigger a per-model retry (transient failures).
    retryable_errors: tuple[type[BaseException], ...] = ()
    #: Exceptions that should trigger fallback to the next model (model exhausted).
    fatal_errors: tuple[type[BaseException], ...] = ()

    @abstractmethod
    def build(self, model_name: str, **overrides: Any) -> BaseChatModel:
        """Construct a chat model for ``model_name`` with optional overrides.

        Args:
            model_name: The provider-specific model identifier.
            **overrides: Generic, provider-agnostic overrides (e.g. ``temperature``,
                ``max_tokens``). The adapter translates/forwards what applies and
                ignores the rest.

        Returns:
            A configured ``BaseChatModel``.
        """
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        """Estimate the token count of ``text``.

        Default is a conservative character-based heuristic (~3 chars/token,
        biased high so message trimming never under-estimates the budget).
        Providers with an exact local tokenizer (e.g. OpenAI/tiktoken) override
        this. Avoids per-call network round-trips on the hot trimming path.
        """
        return len(text) // 3 + 1 if text else 0
