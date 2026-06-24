"""Dialect-aware mem0 vector store configuration for long-term memory.

pgvector lives inside PostgreSQL, so MySQL deployments use a standalone vector
store (Weaviate). Only the mem0 ``vector_store`` block changes between
providers — the ``llm`` and ``embedder`` blocks are owned by MemoryService.
"""

from typing import Any

from agent.core.config import settings


def get_vector_store_config() -> dict[str, Any]:
    """Return the mem0 ``vector_store`` config block for the active provider.

    Returns:
        A dict with ``provider`` and ``config`` keys suitable for splicing into
        ``AsyncMemory.from_config``.

    Raises:
        ValueError: If ``settings.vector.provider`` is not supported.
    """
    provider = settings.vector.provider

    if provider == "weaviate":
        return {
            "provider": "weaviate",
            "config": {
                "collection_name": settings.memory.collection_name,
                "cluster_url": settings.vector.weaviate_cluster_url,
                "auth_client_secret": settings.vector.weaviate_api_key.get_secret_value() or None,
            },
        }

    if provider == "pgvector":
        return {
            "provider": "pgvector",
            "config": {
                "collection_name": settings.memory.collection_name,
                "dbname": settings.database.name,
                "user": settings.database.user,
                "password": settings.database.password.get_secret_value(),
                "host": settings.database.host,
                "port": settings.database.port,
            },
        }

    raise ValueError(f"unsupported VECTOR_STORE_PROVIDER: {provider!r} (expected 'pgvector' or 'weaviate')")
