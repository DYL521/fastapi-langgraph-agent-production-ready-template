"""App-scoped service container.

All services are lazily built via ``cached_property`` so the dependency graph
is resolved on first access, and every subsequent access returns the same
instance.  The container is created once in the FastAPI lifespan, stored on
``app.state.container``, and exposed to route handlers via ``deps.py``.
"""

from functools import cached_property

from langfuse.langchain import CallbackHandler

from agent.core.cache import (
    CacheService,
    create_cache_service,
)
from agent.core.langgraph.graph import LangGraphAgent
from agent.core.logging import logger
from agent.services.database import Database
from agent.services.llm import LLMService
from agent.services.memory import MemoryService


class AppContainer:
    """Owns every app-scoped service; built once at startup, closed at shutdown."""

    @cached_property
    def database(self) -> Database:  # noqa: D102
        return Database()

    @cached_property
    def cache_service(self) -> CacheService:  # noqa: D102
        return create_cache_service()

    @cached_property
    def langfuse_handler(self) -> CallbackHandler:  # noqa: D102
        return CallbackHandler()

    @cached_property
    def llm_service(self) -> LLMService:  # noqa: D102
        return LLMService()

    @cached_property
    def memory_service(self) -> MemoryService:  # noqa: D102
        return MemoryService(cache=self.cache_service)

    @cached_property
    def agent(self) -> LangGraphAgent:  # noqa: D102
        return LangGraphAgent(
            llm_service=self.llm_service,
            memory_service=self.memory_service,
            langfuse_handler=self.langfuse_handler,
        )

    async def startup(self) -> None:
        """Warm services that need async initialization."""
        try:
            await self.cache_service.initialize()
        except Exception as e:
            logger.exception("cache_initialization_failed", error=str(e))

        try:
            await self.agent.create_graph()
            logger.info("graph_pre_warmed")
        except Exception as e:
            logger.exception("graph_pre_warm_failed", error=str(e))

        try:
            await self.memory_service.initialize()
        except Exception as e:
            logger.exception("memory_service_pre_warm_failed", error=str(e))

    async def shutdown(self) -> None:
        """Release resources held by app-scoped services."""
        await self.agent.close()
        await self.cache_service.close()
        await self.database.dispose()
        logger.info("container_shutdown_complete")
