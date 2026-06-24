"""Dialect-aware LangGraph checkpointer, connection pool, and checkpoint cleanup.

Encapsulates the PostgreSQL (psycopg + AsyncPostgresSaver) and MySQL
(aiomysql + AIOMySQLSaver) paths so ``graph.py`` never imports a driver directly.

Import note: psycopg / langgraph-checkpoint-postgres are default dependencies and
are imported at module top. The MySQL driver (aiomysql) and
langgraph-checkpoint-mysql ship in the optional ``mysql`` extra and may be absent
on a postgres-only install, so those imports are intentionally lazy (inside the
MySQL branches) and ``# type: ignore``'d — this is a deliberate exception to the
"imports at top" rule to avoid importing an uninstalled optional backend.
"""

from typing import Any
from urllib.parse import quote_plus

from psycopg import (
    AsyncConnection,
    sql,
)
from psycopg.rows import (
    DictRow,
    dict_row,
)
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agent.core.config import settings

PostgresConnPool = AsyncConnectionPool[AsyncConnection[DictRow]]


async def create_checkpointer_pool() -> Any:
    """Create and open the async connection pool for the configured dialect.

    Returns:
        An open psycopg ``AsyncConnectionPool`` (postgres) or an aiomysql
        ``Pool`` (mysql). Typed as ``Any`` because the two pool types are
        unrelated; callers pass it straight back to the matching factory.
    """
    if settings.database.dialect == "mysql":
        return await _create_mysql_pool()
    return await _create_postgres_pool()


async def _create_postgres_pool() -> PostgresConnPool:
    connection_url = (
        "postgresql://"
        f"{quote_plus(settings.database.user)}:{quote_plus(settings.database.password.get_secret_value())}"
        f"@{settings.database.host}:{settings.database.port}/{settings.database.name}"
    )
    pool: PostgresConnPool = AsyncConnectionPool(
        connection_url,
        open=False,
        max_size=settings.database.pool_size,
        kwargs={
            "autocommit": True,
            "connect_timeout": 5,
            "prepare_threshold": None,
            "row_factory": dict_row,
        },
    )
    await pool.open()
    return pool


async def _create_mysql_pool() -> Any:
    import aiomysql  # type: ignore[import-not-found]

    return await aiomysql.create_pool(
        host=settings.database.host,
        port=settings.database.port,
        user=settings.database.user,
        password=settings.database.password.get_secret_value(),
        db=settings.database.name,
        charset="utf8mb4",  # default is latin1; must match the utf8mb4 columns
        autocommit=True,  # required by AIOMySQLSaver
        minsize=1,
        maxsize=settings.database.pool_size,
        connect_timeout=5,
    )


def create_checkpointer(pool: Any) -> Any:
    """Wrap an open pool in the dialect-appropriate LangGraph saver.

    The caller is responsible for invoking ``await checkpointer.setup()`` once.
    """
    if settings.database.dialect == "mysql":
        from langgraph.checkpoint.mysql.aio import AIOMySQLSaver  # type: ignore[import-not-found]

        return AIOMySQLSaver(conn=pool)
    return AsyncPostgresSaver(pool)


async def delete_thread_checkpoints(pool: Any, thread_id: str) -> None:
    """Delete all checkpoint rows for a thread across the checkpoint tables.

    Table names come from the fixed ``settings.database.checkpoint_tables`` whitelist, so
    interpolating them into the statement carries no injection risk. The
    ``thread_id`` value is always parameterized via the ``%s`` placeholder, which
    both psycopg and aiomysql accept.
    """
    if settings.database.dialect == "mysql":
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                for table in settings.database.checkpoint_tables:
                    await cur.execute(f"DELETE FROM `{table}` WHERE thread_id = %s", (thread_id,))
        return

    # postgres (psycopg) — batch all DELETEs in a single pipeline round-trip
    async with pool.connection() as conn:
        async with conn.pipeline():
            for table in settings.database.checkpoint_tables:
                await conn.execute(
                    sql.SQL("DELETE FROM {} WHERE thread_id = %s").format(sql.Identifier(table)),
                    (thread_id,),
                )
