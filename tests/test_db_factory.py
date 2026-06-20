"""Unit tests for the dialect-aware database factory layer (src/agent/core/db)."""

import asyncio

import pytest

from agent.core.config import settings
from agent.core.db import (
    build_sqlalchemy_url,
    create_checkpointer,
    get_vector_store_config,
)


@pytest.fixture
def set_attrs(monkeypatch):
    """Override attributes on a nested settings sub-model for one test."""

    def _apply(target, **overrides):
        for key, value in overrides.items():
            monkeypatch.setattr(target, key, value)

    return _apply


# --- build_sqlalchemy_url -------------------------------------------------


@pytest.mark.parametrize(
    ("dialect", "async_", "expected_prefix"),
    [
        ("postgres", False, "postgresql+psycopg2://"),
        ("postgres", True, "postgresql+psycopg://"),
        ("mysql", False, "mysql+pymysql://"),
        ("mysql", True, "mysql+asyncmy://"),
    ],
)
def test_build_url_driver_prefix(set_attrs, dialect, async_, expected_prefix):
    set_attrs(settings.database, dialect=dialect, user="u", password="p", host="h", port=1234, name="d")
    expected = f"{expected_prefix}u:p@h:1234/d"
    if dialect == "mysql":
        expected += "?charset=utf8mb4"  # MySQL connections must use utf8mb4
    assert build_sqlalchemy_url(async_=async_) == expected


def test_build_url_escapes_credentials(set_attrs):
    set_attrs(settings.database, dialect="postgres", user="user@x", password="p@ss:w/rd", host="h", port=5432, name="d")
    url = build_sqlalchemy_url(async_=False)
    assert "user%40x" in url
    assert "p%40ss%3Aw%2Frd" in url


def test_build_url_unsupported_dialect(set_attrs):
    set_attrs(settings.database, dialect="oracle")
    with pytest.raises(ValueError, match="unsupported DB_DIALECT"):
        build_sqlalchemy_url()


# --- get_vector_store_config ---------------------------------------------


def test_vector_pgvector(set_attrs):
    set_attrs(settings.vector, provider="pgvector")
    set_attrs(settings.database, name="d", user="u", password="p", host="h", port=5432)
    set_attrs(settings.memory, collection_name="mem")
    cfg = get_vector_store_config()
    assert cfg["provider"] == "pgvector"
    assert cfg["config"]["dbname"] == "d"
    assert cfg["config"]["host"] == "h"
    assert cfg["config"]["port"] == 5432
    assert cfg["config"]["collection_name"] == "mem"


def test_vector_weaviate_anonymous(set_attrs):
    set_attrs(settings.vector, provider="weaviate", weaviate_cluster_url="http://w:8080", weaviate_api_key="")
    set_attrs(settings.memory, collection_name="Mem")
    cfg = get_vector_store_config()
    assert cfg["provider"] == "weaviate"
    assert cfg["config"]["cluster_url"] == "http://w:8080"
    assert cfg["config"]["auth_client_secret"] is None


def test_vector_weaviate_with_key(set_attrs):
    set_attrs(settings.vector, provider="weaviate", weaviate_cluster_url="http://w:8080", weaviate_api_key="secret")
    set_attrs(settings.memory, collection_name="Mem")
    assert get_vector_store_config()["config"]["auth_client_secret"] == "secret"


def test_vector_unsupported_provider(set_attrs):
    set_attrs(settings.vector, provider="faiss")
    with pytest.raises(ValueError, match="unsupported VECTOR_STORE_PROVIDER"):
        get_vector_store_config()


# --- create_checkpointer dialect selection -------------------------------


def test_create_checkpointer_postgres(set_attrs):
    set_attrs(settings.database, dialect="postgres")
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    # The saver's __init__ captures the running loop, so build inside one.
    saver = asyncio.run(_build_checkpointer())
    assert isinstance(saver, AsyncPostgresSaver)


def test_create_checkpointer_mysql_selects_aiomysql(set_attrs):
    set_attrs(settings.database, dialect="mysql")
    pytest.importorskip("langgraph.checkpoint.mysql.aio")
    from langgraph.checkpoint.mysql.aio import AIOMySQLSaver

    saver = asyncio.run(_build_checkpointer())
    assert isinstance(saver, AIOMySQLSaver)


async def _build_checkpointer():
    return create_checkpointer(pool=object())
