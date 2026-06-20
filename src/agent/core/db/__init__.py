"""Dialect-aware database factories.

Centralizes every place the application chooses between PostgreSQL and MySQL so
the rest of the codebase never hardcodes a driver or a ``postgresql://`` prefix.
The active dialect is selected by ``settings.database.dialect`` ("postgres" | "mysql").
"""

from agent.core.db.checkpointer import (
    create_checkpointer,
    create_checkpointer_pool,
    delete_thread_checkpoints,
)
from agent.core.db.url import build_sqlalchemy_url
from agent.core.db.vector import get_vector_store_config

__all__ = [
    "build_sqlalchemy_url",
    "create_checkpointer",
    "create_checkpointer_pool",
    "delete_thread_checkpoints",
    "get_vector_store_config",
]
