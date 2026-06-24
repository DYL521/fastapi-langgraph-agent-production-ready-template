"""Unit tests for the LLM provider abstraction and config-driven fallback."""

import asyncio

import pytest
from langchain_openai import ChatOpenAI
from openai import (
    OpenAIError,
    RateLimitError,
)

from agent.core.config import settings
from agent.services.llm import providers
from agent.services.llm import service as svc


@pytest.fixture(autouse=True)
def _clear_provider_cache():
    """Reset the cached active provider around each test."""
    providers.get_active_provider.cache_clear()
    yield
    providers.get_active_provider.cache_clear()


# --- provider factory -----------------------------------------------------


def test_build_chat_model_openai():
    assert isinstance(providers.build_chat_model("gpt-5-mini"), ChatOpenAI)


def test_openai_provider_error_tuples():
    provider = providers.get_active_provider()
    assert RateLimitError in provider.retryable_errors
    assert provider.fatal_errors == (OpenAIError,)


def test_unsupported_provider_raises(monkeypatch):
    monkeypatch.setattr(settings.llm, "provider", "cohere")
    providers.get_active_provider.cache_clear()
    with pytest.raises(ValueError, match="unsupported LLM_PROVIDER"):
        providers.get_active_provider()


def test_anthropic_provider_selected(monkeypatch):
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setattr(settings.llm, "provider", "anthropic")
    providers.get_active_provider.cache_clear()
    assert type(providers.get_active_provider()).__name__ == "AnthropicProvider"


# --- config model chain ---------------------------------------------------


def test_model_chain_dedup(monkeypatch):
    monkeypatch.setattr(settings.llm, "model", "m1")
    monkeypatch.setattr(settings.llm, "fallback_models", ["m1", "m2", "m2", "m3"])
    assert settings.llm.model_chain == ["m1", "m2", "m3"]


# --- fallback loop (provider-agnostic error handling) ---------------------


class _FakeLLM:
    def __init__(self, exc=None, value=None):
        self._exc = exc
        self._value = value

    async def ainvoke(self, _messages):
        if self._exc is not None:
            raise self._exc
        return self._value


class _FakeProvider:
    retryable_errors = ()
    fatal_errors = (ValueError,)


def test_fallback_loop_returns_on_success(monkeypatch):
    monkeypatch.setattr(svc, "get_active_provider", lambda: _FakeProvider())
    service = svc.LLMService()
    fake = _FakeLLM(value="ok")
    result = asyncio.run(service._fallback_loop(["hi"], 0, lambda _i: fake, lambda _i: None))
    assert result == "ok"


def test_fallback_loop_raises_after_fatal(monkeypatch):
    monkeypatch.setattr(svc, "get_active_provider", lambda: _FakeProvider())
    service = svc.LLMService()
    fake = _FakeLLM(exc=ValueError("boom"))
    with pytest.raises(RuntimeError, match="failed to get response"):
        asyncio.run(service._fallback_loop(["hi"], 0, lambda _i: fake, lambda _i: None))
