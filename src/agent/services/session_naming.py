"""Session auto-naming feature.

On the first message of a new session this module:
  1. Atomically claims the session (prevents duplicate LLM calls across
     concurrent requests and multiple uvicorn workers) via
     ``SessionRepository.claim_name``.
  2. Writes a placeholder name derived from the user's message so the session
     always has a sensible name even if the LLM call later fails.
  3. Fires a background asyncio task that calls a fast nano model with
     structured output to generate a proper title and overwrites the placeholder.
"""

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from agent.core.logging import logger
from agent.core.metrics import session_names_generated_total
from agent.core.prompts import SESSION_TITLE_PROMPT
from agent.repositories import session_repository
from agent.schemas.chat import SessionTitle
from agent.services.llm import llm_service
from agent.utils import spawn_background_task

_PLACEHOLDER_MAX = 40


def _build_placeholder(user_message: str) -> str:
    cleaned = " ".join(user_message.split())
    return cleaned[:_PLACEHOLDER_MAX].rstrip() or "New chat"


async def _persist_session_name(session_id: str, user_message: str) -> None:
    try:
        result = await llm_service.call(
            [
                SystemMessage(content=SESSION_TITLE_PROMPT),
                HumanMessage(content=user_message[:500]),
            ],
            model_name="gpt-5.4-nano",
            response_format=SessionTitle,
            reasoning={"effort": "low"},
            max_tokens=32,
            temperature=0.3,
        )
        await session_repository.update_name(session_id, result.title)
        session_names_generated_total.labels(status="success").inc()
        logger.info("session_name_generated", session_id=session_id, name=result.title)
    except Exception:
        session_names_generated_total.labels(status="error").inc()
        logger.exception("session_name_generation_failed", session_id=session_id)


def maybe_name_session(session_id: str, session_name: str, messages: list) -> None:
    """Trigger session auto-naming if the session is still unnamed.

    Safe to call from any chat endpoint — concurrent callers for the same
    session are deduplicated by the atomic claim.
    """
    if session_name:
        return
    first_user_msg = next((m.content for m in messages if m.role == "user"), None)
    if not first_user_msg:
        return
    if session_repository.claim_name(session_id, _build_placeholder(first_user_msg)):
        spawn_background_task(_persist_session_name(session_id, first_user_msg))
