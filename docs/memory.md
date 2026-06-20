# Memory

## Overview

The template includes a long-term memory system powered by [mem0](https://github.com/mem0ai/mem0) and a configurable vector store. Memories are extracted from conversations, stored as vector embeddings, and retrieved semantically on each request — giving the agent context from past sessions.

The vector store is selected by `VECTOR_STORE_PROVIDER`: **pgvector** (default, shares the PostgreSQL instance) or **Weaviate** (a standalone service used when running on MySQL, since pgvector requires PostgreSQL). The provider is resolved by `get_vector_store_config()` in `src/agent/core/db/vector.py`; only mem0's `vector_store` block changes — the LLM and embedder config stay the same. See [database.md](database.md#database-backends-postgresql--mysql) for the backend matrix.

## How it works

```mermaid
sequenceDiagram
    participant G as LangGraph
    participant MS as MemoryService
    participant Cache as Cache (Valkey/TTL)
    participant M as mem0
    participant PG as pgvector

    Note over G: On every chat request
    G->>MS: search(user_id, query)
    MS->>Cache: get(memory:{user_id}:{hash})
    alt cache hit
        Cache-->>MS: cached result
    else cache miss
        MS->>M: memory.search(user_id, query)
        M->>PG: vector similarity search
        PG-->>M: top-k memories
        M-->>MS: formatted results
        MS->>Cache: set(key, result, TTL)
    end
    MS-->>G: relevant memories string

    Note over G: After LLM response (background)
    G-)MS: add(user_id, messages)
    MS->>M: memory.add(messages, user_id)
    M->>PG: store new embeddings
```

## Cache layer

Memory search results are cached to avoid repeated pgvector queries for similar questions within the same TTL window.

- **With Valkey/Redis**: cache is shared across app instances. Set `VALKEY_HOST` in your `.env`.
- **Without Valkey**: falls back to an in-memory `TTLCache` — works fine for single instances.

Cache key: `memory:{user_id}:{sha256(query)[:16]}`
TTL: `CACHE_TTL_SECONDS` (default: 60s)

Only successful, non-empty results are cached. Errors are never cached.

## Memory updates

After the LLM produces a response, memories are updated **in the background** via `asyncio.create_task`. This means:
- The response is returned immediately, without waiting for mem0 to finish
- Memory updates don't block or slow down the chat response

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `VECTOR_STORE_PROVIDER` | `pgvector` (`weaviate` when `DB_DIALECT=mysql`) | Vector store backend |
| `WEAVIATE_CLUSTER_URL` | `http://localhost:8080` | Weaviate endpoint (when provider is `weaviate`) |
| `WEAVIATE_API_KEY` | _(empty)_ | Weaviate auth key; empty = anonymous |
| `LONG_TERM_MEMORY_COLLECTION_NAME` | `longterm_memory` | Vector collection name |
| `LONG_TERM_MEMORY_MODEL` | `gpt-5-nano` | LLM used by mem0 to extract and process memories |
| `LONG_TERM_MEMORY_EMBEDDER_MODEL` | `text-embedding-3-small` | Embedding model for semantic search |
| `CACHE_TTL_SECONDS` | `60` | Memory search cache TTL |

## Startup pre-warming

At startup, `memory_service.initialize()` is called in the app lifespan. This establishes the pgvector connection pool and runs mem0's schema check, so the first user request doesn't pay the ~130ms cold-init cost.

## Per-user isolation

Each user's memories are stored and searched independently using `user_id` as the namespace. Users cannot access each other's memories.
