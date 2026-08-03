"""LLM service with retries, circular fallback, and optional structured output."""

import asyncio
import logging
from typing import (
    Any,
    Callable,
    TypeVar,
    overload,
)

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agent.core.config import settings
from agent.core.logging import logger
from agent.services.llm.providers import (
    build_chat_model,
    get_active_provider,
)
from agent.services.llm.registry import LLMRegistry

T = TypeVar("T", bound=BaseModel)


class LLMService:
    """Service for managing LLM calls with retries and circular fallback.

    Two distinct execution paths:

    - **Default path** (no model_name / response_format / model_kwargs): uses
      ``self._llm`` which is the tool-bound agent model. Circular fallback
      updates ``self._llm`` so tool bindings are preserved across retries.

    - **One-off path** (any override provided): resolves a fresh, local
      ``Runnable`` for the call without ever touching ``self._llm``, so
      concurrent default-path calls are never affected.
    """

    def __init__(self):
        """Initialize the LLM service with the configured default model."""
        self._llm: Any = None
        self._current_model_index: int = 0
        self._bound_tools: list = []

        all_names = LLMRegistry.get_all_names()
        try:
            self._current_model_index = all_names.index(settings.llm.model)
            self._llm = LLMRegistry.get(settings.llm.model)
            logger.info(
                "llm_service_initialized",
                default_model=settings.llm.model,
                model_index=self._current_model_index,
                total_models=len(all_names),
                environment=settings.app.environment.value,
            )
        except Exception as e:
            self._current_model_index = 0
            self._llm = LLMRegistry.LLMS[0]["llm"]
            logger.warning(
                "default_model_not_found_using_first",
                requested=settings.llm.model,
                using=all_names[0] if all_names else "none",
                error=str(e),
            )

    @overload
    async def call(
        self,
        messages: LanguageModelInput,
        model_name: str | None = ...,
        response_format: None = ...,
        **model_kwargs: Any,
    ) -> BaseMessage: ...

    @overload
    async def call(
        self,
        messages: LanguageModelInput,
        model_name: str | None = ...,
        *,
        response_format: type[T],
        **model_kwargs: Any,
    ) -> T: ...

    async def call(
        self,
        messages: LanguageModelInput,
        model_name: str | None = None,
        response_format: type[BaseModel] | None = None,
        **model_kwargs: Any,
    ) -> BaseMessage | BaseModel:
        """Call the LLM with retries and circular fallback."""
        try:
            return await asyncio.wait_for(
                self._call_with_fallback(messages, model_name, response_format, model_kwargs),
                timeout=settings.llm.total_timeout,
            )
        except asyncio.TimeoutError:
            logger.exception(
                "llm_total_timeout_exceeded",
                timeout_seconds=settings.llm.total_timeout,
            )
            raise RuntimeError(f"llm call timed out after {settings.llm.total_timeout}s total budget")

    def get_llm(self) -> Any:
        """Return the current tool-bound default LLM instance."""
        return self._llm

    def bind_tools(self, tools: list) -> "LLMService":
        """Bind tools to the default LLM instance."""
        if self._llm:
            self._bound_tools = tools
            self._llm = self._llm.bind_tools(tools)
            logger.debug("tools_bound_to_llm", tool_count=len(tools))
        return self

    async def _invoke_with_retry(self, llm: Any, messages: LanguageModelInput) -> Any:
        retryer = AsyncRetrying(
            stop=stop_after_attempt(settings.llm.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(get_active_provider().retryable_errors),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        response = await retryer(llm.ainvoke, messages)
        logger.debug("llm_call_successful")
        return response

    def _switch_to_next_model(self) -> bool:
        try:
            next_index = (self._current_model_index + 1) % len(LLMRegistry.LLMS)
            next_entry = LLMRegistry.get_model_at_index(next_index)
            logger.warning(
                "switching_to_next_model",
                from_index=self._current_model_index,
                to_index=next_index,
                to_model=next_entry["name"],
            )
            self._current_model_index = next_index
            self._llm = next_entry["llm"]
            if self._bound_tools:
                self._llm = self._llm.bind_tools(self._bound_tools)
            logger.info("model_switched", new_model=next_entry["name"], new_index=next_index)
            return True
        except Exception:
            logger.exception("model_switch_failed")
            return False

    async def _call_with_fallback(
        self,
        messages: LanguageModelInput,
        model_name: str | None,
        response_format: type[BaseModel] | None,
        model_kwargs: dict,
    ) -> BaseMessage | BaseModel:
        def _override_target(idx: int) -> Any:
            base = LLMRegistry.get(LLMRegistry.LLMS[idx]["name"], **model_kwargs)
            return base.with_structured_output(response_format) if response_format else base

        def _default_target(_: int) -> Any:
            return self._llm

        def _default_advance(_: int) -> int | None:
            return self._current_model_index if self._switch_to_next_model() else None

        if model_name or response_format or model_kwargs:
            all_names = LLMRegistry.get_all_names()
            if model_name and model_name not in all_names:
                logger.debug("building_out_of_chain_model", model_name=model_name)
                target = build_chat_model(model_name, **model_kwargs)
                if response_format:
                    target = target.with_structured_output(response_format)
                return await self._invoke_with_retry(target, messages)

            start = all_names.index(model_name) if model_name else self._current_model_index
            total = len(LLMRegistry.LLMS)
            get_target: Callable[[int], Any] = _override_target

            def _override_advance(idx: int) -> int | None:
                return (idx + 1) % total

            advance: Callable[[int], int | None] = _override_advance
        else:
            start = self._current_model_index
            get_target = _default_target
            advance = _default_advance

        return await self._fallback_loop(messages, start, get_target, advance)

    async def _fallback_loop(
        self,
        messages: LanguageModelInput,
        start: int,
        get_target: Callable[[int], Any],
        advance: Callable[[int], int | None],
    ) -> Any:
        fatal_errors = get_active_provider().fatal_errors or (Exception,)
        total = len(LLMRegistry.LLMS)
        current = start
        models_tried = 0
        last_error: BaseException | None = None

        for models_tried in range(1, total + 1):
            current_name = LLMRegistry.LLMS[current]["name"]
            try:
                return await self._invoke_with_retry(get_target(current), messages)
            except fatal_errors as e:
                last_error = e
                logger.error(
                    "llm_call_failed_after_retries",
                    model=current_name,
                    models_tried=models_tried,
                    total_models=total,
                    error=str(e),
                )
                if models_tried >= total:
                    logger.error(
                        "all_models_failed", models_tried=models_tried, starting_model=LLMRegistry.LLMS[start]["name"]
                    )
                    break
                next_idx = advance(current)
                if next_idx is None:
                    logger.error("failed_to_switch_to_next_model")
                    break
                current = next_idx

        raise RuntimeError(
            f"failed to get response from llm after trying {models_tried} models. last error: {str(last_error)}"
        )


