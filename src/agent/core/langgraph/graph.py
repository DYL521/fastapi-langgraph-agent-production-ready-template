"""This file contains the LangGraph Agent/workflow and interactions with the LLM."""

import asyncio
from typing import (
    Any,
    AsyncGenerator,
    cast,
)

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langgraph.errors import GraphInterrupt
from langgraph.graph import (
    END,
    StateGraph,
)
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import (
    Command,
    CompiledStateGraph,
)
from langgraph.types import (
    RetryPolicy,
    StateSnapshot,
)

from agent.core.config import (
    Environment,
    settings,
)
from agent.core.db import (
    create_checkpointer,
    create_checkpointer_pool,
    delete_thread_checkpoints,
)
from agent.core.langgraph.tools import tools
from agent.core.logging import logger
from agent.core.metrics import llm_inference_duration_seconds
from agent.core.observability import langfuse_callback_handler
from agent.core.prompts import load_system_prompt
from agent.schemas import (
    GraphState,
    Message,
)
from agent.services.llm import LLMService
from agent.services.memory import memory_service
from agent.utils import (
    dump_messages,
    extract_text_content,
    prepare_messages,
    process_llm_response,
    spawn_background_task,
)

class LangGraphAgent:
    """Manages the LangGraph Agent/workflow and interactions with the LLM."""

    def __init__(self, llm_service: LLMService):
        """Initialize the LangGraph Agent with necessary components."""
        self.llm_service = llm_service
        self.llm_service.bind_tools(tools)
        self.tools_by_name = {tool.name: tool for tool in tools}
        self._connection_pool: Any | None = None
        self._graph: CompiledStateGraph | None = None
        logger.info(
            "langgraph_agent_initialized",
            model=settings.llm.model,
            environment=settings.app.environment.value,
        )

    def _build_config(
        self,
        session_id: str,
        user_id: str | None = None,
        username: str | None = None,
    ) -> RunnableConfig:
        callbacks: list[BaseCallbackHandler] = [langfuse_callback_handler] if settings.langfuse.tracing_enabled else []
        return {
            "configurable": {"thread_id": session_id},
            "callbacks": callbacks,
            "metadata": {
                "user_id": user_id,
                "username": username,
                "session_id": session_id,
                "environment": settings.app.environment.value,
                "debug": settings.app.debug,
            },
        }

    async def _prepare_input(
        self,
        graph: CompiledStateGraph,
        config: RunnableConfig,
        messages: list[Message],
        user_id: str | None,
    ) -> tuple[StateSnapshot, Any]:
        """Run state check and memory search concurrently, return (state, graph_input)."""
        state, relevant_memory = await asyncio.gather(
            graph.aget_state(config),
            memory_service.search(user_id, messages[-1].content),
        )

        if state.next:
            return state, Command(resume=messages[-1].content)

        relevant_memory = relevant_memory or "No relevant memory found."
        return state, {"messages": dump_messages(messages), "long_term_memory": relevant_memory}

    async def _get_connection_pool(self) -> Any | None:
        if self._connection_pool is None:
            try:
                self._connection_pool = await create_checkpointer_pool()
                logger.info(
                    "connection_pool_created",
                    max_size=settings.database.pool_size,
                    dialect=settings.database.dialect,
                    environment=settings.app.environment.value,
                )
            except Exception as e:
                logger.exception("connection_pool_creation_failed", environment=settings.app.environment.value)
                if settings.app.environment == Environment.PRODUCTION:
                    logger.warning("continuing_without_connection_pool", environment=settings.app.environment.value)
                    return None
                raise e
        return self._connection_pool

    async def _chat(self, state: GraphState, config: RunnableConfig) -> Command:
        current_llm = self.llm_service.get_llm()
        model_name = (
            current_llm.model_name
            if current_llm and hasattr(current_llm, "model_name")
            else settings.llm.model
        )

        username = config.get("metadata", {}).get("username")
        thread_id = config.get("configurable", {}).get("thread_id")
        SYSTEM_PROMPT = load_system_prompt(username=username, long_term_memory=state.long_term_memory)

        messages = prepare_messages(state.messages, SYSTEM_PROMPT)

        try:
            with llm_inference_duration_seconds.labels(model=model_name).time():
                response_message = await self.llm_service.call(dump_messages(messages))

            response_message = process_llm_response(response_message)

            logger.info(
                "llm_response_generated",
                session_id=thread_id,
                model=model_name,
                environment=settings.app.environment.value,
            )

            if isinstance(response_message, AIMessage) and response_message.tool_calls:
                goto = "tool_call"
            else:
                goto = END

            return Command(update={"messages": [response_message]}, goto=goto)
        except Exception:
            logger.exception(
                "llm_call_failed_all_models",
                session_id=thread_id,
                environment=settings.app.environment.value,
            )
            raise

    async def _tool_call(self, state: GraphState) -> Command:
        tool_calls = state.messages[-1].tool_calls

        async def _execute_tool(tool_call: dict) -> ToolMessage:
            tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
            return ToolMessage(
                content=tool_result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )

        if len(tool_calls) == 1:
            outputs = [await _execute_tool(tool_calls[0])]
        else:
            outputs = list(await asyncio.gather(*[_execute_tool(tc) for tc in tool_calls]))

        return Command(update={"messages": outputs}, goto="chat")

    async def create_graph(self) -> CompiledStateGraph | None:
        """Create and configure the LangGraph workflow."""
        if self._graph is None:
            try:
                graph_builder = StateGraph(GraphState)
                graph_builder.add_node("chat", self._chat, destinations=("tool_call", END))
                graph_builder.add_node(
                    "tool_call",
                    self._tool_call,
                    destinations=("chat",),
                    retry_policy=RetryPolicy(max_attempts=3),
                )
                graph_builder.set_entry_point("chat")
                graph_builder.set_finish_point("chat")

                connection_pool = await self._get_connection_pool()
                if connection_pool:
                    checkpointer = create_checkpointer(connection_pool)
                    await checkpointer.setup()
                else:
                    checkpointer = None
                    if settings.app.environment != Environment.PRODUCTION:
                        raise Exception("Connection pool initialization failed")

                self._graph = graph_builder.compile(
                    checkpointer=checkpointer, name=f"{settings.app.project_name} Agent ({settings.app.environment.value})"
                )

                logger.info(
                    "graph_created",
                    graph_name=f"{settings.app.project_name} Agent",
                    environment=settings.app.environment.value,
                    has_checkpointer=checkpointer is not None,
                )
            except Exception as e:
                logger.exception("graph_creation_failed", environment=settings.app.environment.value)
                if settings.app.environment == Environment.PRODUCTION:
                    logger.warning("continuing_without_graph")
                    return None
                raise e

        return self._graph

    async def _get_graph(self) -> CompiledStateGraph:
        if self._graph is None:
            self._graph = await self.create_graph()
        if self._graph is None:
            raise RuntimeError("graph initialization failed")
        return self._graph

    async def get_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: str | None = None,
        username: str | None = None,
    ) -> list[Message]:
        """Get a non-streaming response from the LLM."""
        graph = await self._get_graph()
        config = self._build_config(session_id, user_id, username)

        try:
            state, graph_input = await self._prepare_input(graph, config, messages, user_id)

            if state.next:
                logger.info("resuming_interrupted_graph", session_id=session_id, next_nodes=state.next)

            response = await graph.ainvoke(graph_input, config=config)

            state = await graph.aget_state(config)
            if state.next:
                interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
                logger.info("graph_interrupted", session_id=session_id, interrupt_value=str(interrupt_value))
                return [Message(role="assistant", content=str(interrupt_value))]

            openai_msgs = cast(list[dict], convert_to_openai_messages(response["messages"]))
            spawn_background_task(memory_service.add(user_id, openai_msgs, config.get("metadata")))
            return self._process_messages(response["messages"])
        except GraphInterrupt:
            state = await graph.aget_state(config)
            interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
            logger.info("graph_interrupted", session_id=session_id, interrupt_value=str(interrupt_value))
            return [Message(role="assistant", content=str(interrupt_value))]
        except Exception as e:
            logger.exception("get_response_failed", error=str(e), session_id=session_id)
            raise

    async def get_stream_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: str | None = None,
        username: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Get a streaming response from the LLM, yielding text chunks."""
        config = self._build_config(session_id, user_id, username)
        graph = await self._get_graph()

        try:
            state, graph_input = await self._prepare_input(graph, config, messages, user_id)

            if state.next:
                logger.info("resuming_interrupted_graph_stream", session_id=session_id, next_nodes=state.next)

            async for token, _ in graph.astream(
                graph_input,
                config,
                stream_mode="messages",
            ):
                if not isinstance(token, (AIMessage, AIMessageChunk)):
                    continue

                text = extract_text_content(token.content)
                if text:
                    yield text

            state = await graph.aget_state(config)
            if state.next:
                interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
                logger.info("graph_interrupted_stream", session_id=session_id, interrupt_value=str(interrupt_value))
                yield str(interrupt_value)
            elif state.values and "messages" in state.values:
                openai_msgs = cast(list[dict], convert_to_openai_messages(state.values["messages"]))
                spawn_background_task(memory_service.add(user_id, openai_msgs, config.get("metadata")))
        except GraphInterrupt:
            state = await graph.aget_state(config)
            interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
            logger.info("graph_interrupted_stream", session_id=session_id, interrupt_value=str(interrupt_value))
            yield str(interrupt_value)
        except Exception as stream_error:
            logger.exception("stream_processing_failed", error=str(stream_error), session_id=session_id)
            raise stream_error

    async def get_chat_history(self, session_id: str, limit: int | None = None) -> list[Message]:
        """Get the chat history for a given session."""
        graph = await self._get_graph()

        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        state: StateSnapshot = await graph.aget_state(config=config)
        if not state.values:
            return []
        messages = self._process_messages(state.values["messages"])
        return messages[-limit:] if limit else messages

    def _process_messages(self, messages: list[BaseMessage]) -> list[Message]:
        openai_style_messages = convert_to_openai_messages(messages)
        return [
            Message(role=message["role"], content=str(message["content"]))
            for message in openai_style_messages
            if message["role"] in ["assistant", "user"] and message["content"]
        ]

    async def clear_chat_history(self, session_id: str) -> None:
        """Clear all chat history for a given session."""
        try:
            conn_pool = await self._get_connection_pool()
            if conn_pool is None:
                raise RuntimeError("connection pool unavailable; cannot clear chat history")

            await delete_thread_checkpoints(conn_pool, session_id)
            logger.info(
                "checkpoint_tables_cleared_for_session",
                tables=settings.database.checkpoint_tables,
                session_id=session_id,
            )

        except Exception:
            logger.exception(
                "clear_chat_history_operation_failed",
                session_id=session_id,
            )
            raise

    async def close(self) -> None:
        """Shut down the agent's resources (connection pool)."""
        if self._connection_pool:
            await self._connection_pool.close()
            self._connection_pool = None
            logger.info("connection_pool_closed")
