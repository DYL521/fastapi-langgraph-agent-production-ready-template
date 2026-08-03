"""LLM model registry — config-driven, provider-agnostic."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from agent.core.config import settings
from agent.core.logging import logger
from agent.services.llm.providers import build_chat_model


def _build_registry() -> list[dict[str, Any]]:
    """Build the registry entries from the configured model chain."""
    return [{"name": name, "llm": build_chat_model(name)} for name in settings.llm.model_chain]


class LLMRegistry:
    """Registry of available LLM models with pre-initialized instances."""

    LLMS: list[dict[str, Any]] = _build_registry()

    @classmethod
    def get(cls, model_name: str, **kwargs) -> BaseChatModel:
        """Get an LLM by name, optionally with per-call overrides."""
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
    def get_all_names(cls) -> list[str]:
        """Return all registered model names in order."""
        return [e["name"] for e in cls.LLMS]

    @classmethod
    def get_model_at_index(cls, index: int) -> dict[str, Any]:
        """Return the model entry at an index, wrapping to 0 if out of range."""
        if 0 <= index < len(cls.LLMS):
            return cls.LLMS[index]
        return cls.LLMS[0]
