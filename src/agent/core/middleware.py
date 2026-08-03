"""Custom middleware for tracking metrics and other cross-cutting concerns."""

import json
import time
import tracemalloc
from typing import TYPE_CHECKING

from asgi_correlation_id import correlation_id
from jose import (
    JWTError,
    jwt,
)
from starlette.types import (
    ASGIApp,
    Message,
    Receive,
    Scope,
    Send,
)

from agent.core.config import settings
from agent.core.logging import (
    bind_context,
    clear_context,
    logger,
)
from agent.core.metrics import (
    http_request_duration_seconds,
    http_requests_total,
)

if TYPE_CHECKING:
    from pyinstrument import Profiler  # pyright: ignore[reportMissingImports]
    from pyinstrument.renderers import JSONRenderer  # pyright: ignore[reportMissingImports]

    PYINSTRUMENT_AVAILABLE = True
else:
    try:
        from pyinstrument import Profiler
        from pyinstrument.renderers import JSONRenderer

        PYINSTRUMENT_AVAILABLE = True
    except ImportError:
        Profiler = None
        JSONRenderer = None
        PYINSTRUMENT_AVAILABLE = False


class MetricsMiddleware:
    """Pure ASGI middleware for tracking HTTP request metrics.

    Avoids BaseHTTPMiddleware overhead — no request body buffering, no
    StreamingResponse backpressure issues.
    """

    def __init__(self, app: ASGIApp):
        """Wrap an ASGI application."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Track request count and duration for HTTP requests."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.time() - start_time
            path = scope.get("path", "")
            method = scope.get("method", "")
            http_requests_total.labels(method=method, endpoint=path, status=status_code).inc()
            http_request_duration_seconds.labels(method=method, endpoint=path).observe(duration)


class LoggingContextMiddleware:
    """Pure ASGI middleware for adding user_id and session_id to logging context."""

    def __init__(self, app: ASGIApp):
        """Wrap an ASGI application."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Extract session_id from JWT and bind to structlog context."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        try:
            clear_context()

            headers = dict(scope.get("headers", []))
            auth_header = headers.get(b"authorization", b"").decode()
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ", 1)[1]
                try:
                    payload = jwt.decode(
                        token, settings.jwt.secret_key.get_secret_value(), algorithms=[settings.jwt.algorithm]
                    )
                    session_id = payload.get("sub")
                    if session_id:
                        bind_context(session_id=session_id)
                except JWTError:
                    pass

            await self.app(scope, receive, send)
        finally:
            clear_context()


class ProfilingMiddleware:
    """Pure ASGI middleware for per-request profiling using pyinstrument.

    Only active when DEBUG=true. Profiles every request and saves a JSON
    report when the request exceeds PROFILING_THRESHOLD_SECONDS.
    """

    def __init__(self, app: ASGIApp):
        """Wrap an ASGI application."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Profile every request; save enriched JSON if duration exceeds threshold."""
        if scope["type"] != "http" or not PYINSTRUMENT_AVAILABLE:
            await self.app(scope, receive, send)
            return

        tracemalloc.start()
        cpu_start = time.process_time()

        profiler = Profiler(async_mode="enabled")
        with profiler:
            await self.app(scope, receive, send)

        cpu_ms = round((time.process_time() - cpu_start) * 1000, 2)
        mem_current_kb, mem_peak_kb = (v // 1024 for v in tracemalloc.get_traced_memory())
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        wall_ms = round((profiler.last_session.duration if profiler.last_session else 0.0) * 1000, 2)

        if wall_ms / 1000 >= settings.logging.profiling_threshold_seconds:
            raw_id = correlation_id.get() or "unknown"
            if len(raw_id) == 32 and "-" not in raw_id:
                raw_id = f"{raw_id[:8]}-{raw_id[8:12]}-{raw_id[12:16]}-{raw_id[16:20]}-{raw_id[20:]}"

            settings.logging.profiling_dir.mkdir(parents=True, exist_ok=True)
            filepath = settings.logging.profiling_dir / f"{raw_id}.json"

            _excluded = ("tracemalloc", "pyinstrument", "<frozen", "logging/__init__")
            top_allocs = [
                {
                    "file": str(stat.traceback[0].filename).replace(str(__file__).rsplit("/", 3)[0] + "/", ""),
                    "line": stat.traceback[0].lineno,
                    "size_kb": round(stat.size / 1024, 2),
                    "count": stat.count,
                }
                for stat in snapshot.statistics("lineno")
                if not any(ex in str(stat.traceback[0].filename) for ex in _excluded)
            ]

            call_tree = json.loads(profiler.output(renderer=JSONRenderer()))
            path = scope.get("path", "")
            method = scope.get("method", "")
            report = {
                "request_id": raw_id,
                "endpoint": f"{method} {path}",
                "wall_time_ms": wall_ms,
                "cpu_time_ms": cpu_ms,
                "io_wait_ms": round(wall_ms - cpu_ms, 2),
                "memory_peak_kb": mem_peak_kb,
                "memory_allocated_kb": mem_current_kb,
                "top_memory_allocators": top_allocs,
                "call_tree": call_tree,
            }
            filepath.write_text(json.dumps(report, indent=2))
            logger.debug(
                "slow_request_profile_saved",
                path=path,
                method=method,
                wall_time_ms=wall_ms,
                cpu_time_ms=cpu_ms,
                memory_peak_kb=mem_peak_kb,
                io_wait_ms=round(wall_ms - cpu_ms, 2),
                profile_file=str(filepath),
            )
