"""LLM service with retries, circular fallback, and optional structured output."""

import asyncio
import logging
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
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
        self._llm: Any = None  # BaseChatModel pre-bind_tools, Runnable after
        self._current_model_index: int = 0
        self._bound_tools: List = []

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @overload
    async def call(
        self,
        messages: LanguageModelInput,
        model_name: Optional[str] = ...,
        response_format: None = ...,
        **model_kwargs: Any,
    ) -> BaseMessage: ...

    @overload
    async def call(
        self,
        messages: LanguageModelInput,
        model_name: Optional[str] = ...,
        *,
        response_format: Type[T],
        **model_kwargs: Any,
    ) -> T: ...

    async def call(
        self,
        messages: LanguageModelInput,
        model_name: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **model_kwargs: Any,
    ) -> Union[BaseMessage, BaseModel]:
        """Call the LLM with retries and circular fallback.

        Args:
            messages: Conversation messages to send.
            model_name: Override the model. ``None`` uses the current default.
            response_format: Pydantic schema for structured output. When
                provided the call chains ``.with_structured_output(schema)``
                and returns a validated instance of that schema instead of a
                raw ``BaseMessage``.
            **model_kwargs: Extra kwargs forwarded to ``LLMRegistry.get`` when
                constructing a one-off model instance (e.g. ``temperature``,
                ``max_tokens``, ``reasoning``).

        Returns:
            ``BaseMessage`` when ``response_format`` is ``None``, otherwise a
            validated instance of ``response_format``.

        Raises:
            RuntimeError: When all models fail after retries or the total
                timeout budget is exceeded.
        """
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
        """Return the current tool-bound default LLM instance.

        Returns:
            Current ``BaseChatModel`` instance or ``None`` if not initialised.
        """
        return self._llm

    def bind_tools(self, tools: List) -> "LLMService":
        """Bind tools to the default LLM instance.

        Args:
            tools: List of tools to bind.

        Returns:
            Self for method chaining.
        """
        if self._llm:
            self._bound_tools = tools
            self._llm = self._llm.bind_tools(tools)
            logger.debug("tools_bound_to_llm", tool_count=len(tools))
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _invoke_with_retry(self, llm: Any, messages: LanguageModelInput) -> Any:
        """Invoke an LLM runnable with automatic per-model retry logic.

        Retries on the active provider's transient error types (declared by its
        adapter), with exponential backoff. Non-retryable errors propagate
        immediately for the fallback loop to handle.

        Args:
            llm: Any LangChain ``Runnable`` (plain model or structured-output chain).
            messages: Messages to send.

        Returns:
            The runnable's response (``BaseMessage`` or a ``BaseModel`` instance).

        Raises:
            Exception: The provider error, propagated after retries are exhausted.
        """
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
        """Advance the default model to the next entry in the registry (circular).

        Mutates ``self._llm`` and ``self._current_model_index`` so tool bindings
        survive model switches on the default agent path.

        Returns:
            ``True`` on success, ``False`` if the switch failed.
        """
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
        model_name: Optional[str],
        response_format: Optional[Type[BaseModel]],
        model_kwargs: dict,
    ) -> Union[BaseMessage, BaseModel]:
        """Build path-specific strategies and delegate to the shared fallback loop.

        One-off path (any override set):
            ``get_target`` builds a fresh registry instance each attempt.
            ``advance`` increments a local index — ``self._llm`` is never touched.

        Default path (no overrides):
            ``get_target`` returns ``self._llm`` (tool-bound).
            ``advance`` calls ``_switch_to_next_model`` so bindings persist.
        """

        def _override_target(idx: int) -> Any:
            base = LLMRegistry.get(LLMRegistry.LLMS[idx]["name"], **model_kwargs)
            return base.with_structured_output(response_format) if response_format else base

        def _default_target(_: int) -> Any:
            return self._llm

        def _default_advance(_: int) -> Optional[int]:
            return self._current_model_index if self._switch_to_next_model() else None

        if model_name or response_format or model_kwargs:
            all_names = LLMRegistry.get_all_names()
            if model_name and model_name not in all_names:
                # Out-of-chain model (e.g. a dedicated session-naming model):
                # build it directly and run a single attempt with retry, no
                # cross-model fallback.
                logger.debug("building_out_of_chain_model", model_name=model_name)
                target = build_chat_model(model_name, **model_kwargs)
                if response_format:
                    target = target.with_structured_output(response_format)
                return await self._invoke_with_retry(target, messages)

            start = all_names.index(model_name) if model_name else self._current_model_index
            total = len(LLMRegistry.LLMS)
            get_target: Callable[[int], Any] = _override_target

            def _override_advance(idx: int) -> Optional[int]:
                return (idx + 1) % total

            advance: Callable[[int], Optional[int]] = _override_advance
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
        advance: Callable[[int], Optional[int]],
    ) -> Any:
        """Shared fallback loop — try each model in turn until one succeeds.

        Args:
            messages: Messages to send.
            start: Registry index to begin from.
            get_target: Returns the ``Runnable`` to invoke for a given index.
            advance: Returns the next index to try, or ``None`` to stop.

        Returns:
            The first successful response.

        Raises:
            RuntimeError: When all models have been exhausted.
        """
        # Treat the active provider's declared fatal errors as fall-back triggers;
        # if a provider declares none, any Exception triggers fallback.
        fatal_errors = get_active_provider().fatal_errors or (Exception,)
        total = len(LLMRegistry.LLMS)
        current = start
        models_tried = 0
        last_error: Optional[BaseException] = None

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


llm_service = LLMService()
