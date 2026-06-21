"""LLM model registry — config-driven, provider-agnostic.

The model list comes from ``settings.llm.model_chain`` (primary model plus any
``LLM_FALLBACK_MODELS``), and instances are built through the active provider's
adapter (``build_chat_model``). No model names or provider classes are hardcoded.
"""

from typing import (
    Any,
    Dict,
    List,
)

from langchain_core.language_models.chat_models import BaseChatModel

from agent.core.config import settings
from agent.core.logging import logger
from agent.services.llm.providers import build_chat_model


def _build_registry() -> List[Dict[str, Any]]:
    """Build the registry entries from the configured model chain."""
    return [{"name": name, "llm": build_chat_model(name)} for name in settings.llm.model_chain]


class LLMRegistry:
    """Registry of available LLM models with pre-initialized instances."""

    LLMS: List[Dict[str, Any]] = _build_registry()

    @classmethod
    def get(cls, model_name: str, **kwargs) -> BaseChatModel:
        """Get an LLM by name, optionally with per-call overrides.

        When kwargs are provided a fresh instance is built with those overrides,
        leaving the shared registry entry untouched.

        Args:
            model_name: Name of the model to retrieve.
            **kwargs: Optional provider-agnostic overrides.

        Returns:
            BaseChatModel instance.

        Raises:
            ValueError: If model_name is not in the configured chain.
        """
        model_entry = next((e for e in cls.LLMS if e["name"] == model_name), None)
        if not model_entry:
            available = ", ".join(e["name"] for e in cls.LLMS)
            raise ValueError(f"model '{model_name}' not found in registry. available models: {available}")

        if kwargs:
            logger.debug("creating_llm_with_custom_args", model_name=model_name, custom_args=list(kwargs.keys()))
            return build_chat_model(model_name, **kwargs)

        logger.debug("using_default_llm_instance", model_name=model_name)
        return model_entry["llm"]

    @classmethod
    def get_all_names(cls) -> List[str]:
        """Return all registered model names in order."""
        return [e["name"] for e in cls.LLMS]

    @classmethod
    def get_model_at_index(cls, index: int) -> Dict[str, Any]:
        """Return the model entry at an index, wrapping to 0 if out of range."""
        if 0 <= index < len(cls.LLMS):
            return cls.LLMS[index]
        return cls.LLMS[0]
