"""Chatbot API endpoints for handling chat interactions."""

import json

from fastapi import (
    APIRouter,
    Depends,
    Query,
    Request,
)
from fastapi.responses import StreamingResponse

from agent.api.v1.dependencies import (
    get_agent,
    get_current_session,
    get_session_repository,
)
from agent.core.config import settings
from agent.core.langgraph.graph import LangGraphAgent
from agent.core.limiter import limiter
from agent.core.logging import logger
from agent.core.metrics import llm_stream_duration_seconds
from agent.models.session import Session
from agent.repositories import SessionRepository
from agent.schemas.chat import (
    ChatRequest,
    ChatResponse,
    StreamResponse,
)
from agent.services.session_naming import maybe_name_session

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.rate_limit.endpoints["chat"][0])
async def chat(
    request: Request,
    chat_request: ChatRequest,
    session: Session = Depends(get_current_session),
    agent: LangGraphAgent = Depends(get_agent),
    session_repo: SessionRepository = Depends(get_session_repository),
):
    """Process a chat request using LangGraph."""
    logger.info(
        "chat_request_received",
        session_id=session.id,
        message_count=len(chat_request.messages),
    )

    if settings.llm.session_naming_enabled:
        await maybe_name_session(agent.llm_service, session_repo, session.id, session.name, chat_request.messages)

    result = await agent.get_response(
        chat_request.messages, session.id, user_id=str(session.user_id), username=session.username
    )

    logger.info("chat_request_processed", session_id=session.id)

    return ChatResponse(messages=result)


@router.post("/chat/stream")
@limiter.limit(settings.rate_limit.endpoints["chat_stream"][0])
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    session: Session = Depends(get_current_session),
    agent: LangGraphAgent = Depends(get_agent),
    session_repo: SessionRepository = Depends(get_session_repository),
):
    """Process a chat request using LangGraph with streaming response."""
    logger.info(
        "stream_chat_request_received",
        session_id=session.id,
        message_count=len(chat_request.messages),
    )

    if settings.llm.session_naming_enabled:
        await maybe_name_session(agent.llm_service, session_repo, session.id, session.name, chat_request.messages)

    async def event_generator():
        try:
            with llm_stream_duration_seconds.labels(model=agent.llm_service.get_llm().get_name()).time():
                async for chunk in agent.get_stream_response(
                    chat_request.messages, session.id, user_id=str(session.user_id), username=session.username
                ):
                    response = StreamResponse(content=chunk, done=False)
                    yield f"data: {json.dumps(response.model_dump(mode='json'))}\n\n"

            final_response = StreamResponse(content="", done=True)
            yield f"data: {json.dumps(final_response.model_dump(mode='json'))}\n\n"

        except Exception as e:
            logger.exception(
                "stream_chat_request_failed",
                session_id=session.id,
                error=str(e),
            )
            error_response = StreamResponse(content="Internal server error", done=True)
            yield f"data: {json.dumps(error_response.model_dump(mode='json'))}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/messages", response_model=ChatResponse)
@limiter.limit(settings.rate_limit.endpoints["messages"][0])
async def get_session_messages(
    request: Request,
    session: Session = Depends(get_current_session),
    agent: LangGraphAgent = Depends(get_agent),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Get the most recent messages for a session."""
    messages = await agent.get_chat_history(session.id, limit=limit)
    return ChatResponse(messages=messages)


@router.delete("/messages")
@limiter.limit(settings.rate_limit.endpoints["messages"][0])
async def clear_chat_history(
    request: Request,
    session: Session = Depends(get_current_session),
    agent: LangGraphAgent = Depends(get_agent),
):
    """Clear all messages for a session."""
    await agent.clear_chat_history(session.id)
    return {"message": "Chat history cleared successfully"}
