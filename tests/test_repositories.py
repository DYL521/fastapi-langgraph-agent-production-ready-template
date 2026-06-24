"""Repository-level tests over an in-memory SQLite database."""

from agent.repositories.session_repository import SessionRepository
from agent.repositories.user_repository import UserRepository


async def test_user_crud(session_maker):
    users = UserRepository(session_maker)
    user = await users.create("x@example.com", "hashed-pw", "name")
    assert user.id is not None

    fetched = await users.get(user.id)
    assert fetched is not None and fetched.email == "x@example.com"

    by_email = await users.get_by_email("x@example.com")
    assert by_email is not None and by_email.id == user.id

    assert await users.delete_by_email("x@example.com") is True
    assert await users.get_by_email("x@example.com") is None
    assert await users.delete_by_email("x@example.com") is False


async def test_session_claim_name_dedup(session_maker):
    users = UserRepository(session_maker)
    sessions = SessionRepository(session_maker)
    user = await users.create("y@example.com", "hashed-pw")
    await sessions.create("sess-1", user.id, name="")

    # Exactly one claim wins; subsequent claims see a non-empty name and lose.
    assert await sessions.claim_name("sess-1", "first") is True
    assert await sessions.claim_name("sess-1", "second") is False

    stored = await sessions.get("sess-1")
    assert stored is not None and stored.name == "first"


async def test_list_for_user_pagination(session_maker):
    users = UserRepository(session_maker)
    sessions = SessionRepository(session_maker)
    user = await users.create("z@example.com", "hashed-pw")
    for i in range(3):
        await sessions.create(f"s-{i}", user.id, name=f"n{i}")

    page = await sessions.list_for_user(user.id, limit=2, offset=0)
    assert len(page) == 2
    rest = await sessions.list_for_user(user.id, limit=2, offset=2)
    assert len(rest) == 1
