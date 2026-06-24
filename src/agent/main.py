"""This file contains the main application entry point."""

from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    Request,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from asgi_correlation_id import (
    CorrelationIdMiddleware,
    correlation_id,
)

from agent.api.v1.api import api_router
from agent.api.v1.chatbot import agent
from agent.core.cache import cache_service
from agent.core.config import (
    Environment,
    settings,
)
from agent.core.limiter import limiter
from agent.core.logging import logger
from agent.core.metrics import setup_metrics
from agent.core.middleware import (
    LoggingContextMiddleware,
    MetricsMiddleware,
    ProfilingMiddleware,
)
from agent.core.observability import langfuse_init
from agent.services.database import database
from agent.services.memory import memory_service

# Load environment variables
load_dotenv()
langfuse_init()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    logger.info(
        "application_startup",
        project_name=settings.app.project_name,
        version=settings.app.version,
        api_prefix=settings.app.api_v1_str,
    )

    # Initialize cache service (connects to Valkey if configured)
    try:
        await cache_service.initialize()
    except Exception as e:
        logger.exception("cache_initialization_failed", error=str(e))

    # Pre-warm the LangGraph agent: create graph + connection pool at startup
    # to avoid cold-start latency on the first request
    try:
        await agent.create_graph()
        logger.info("graph_pre_warmed")
    except Exception as e:
        logger.exception("graph_pre_warm_failed", error=str(e))

    # Pre-warm mem0 AsyncMemory: initializes pgvector connection and schema check
    # so the first search() cache miss or add() doesn't pay the ~130ms cold-init cost
    try:
        await memory_service.initialize()
    except Exception as e:
        logger.exception("memory_service_pre_warm_failed", error=str(e))

    yield

    # Cleanup on shutdown
    await cache_service.close()
    if agent._connection_pool:
        await agent._connection_pool.close()
        logger.info("connection_pool_closed")
    await database.dispose()
    logger.info("application_shutdown")


app = FastAPI(
    title=settings.app.project_name,
    version=settings.app.version,
    description=settings.app.description,
    openapi_url=f"{settings.app.api_v1_str}/openapi.json",
    lifespan=lifespan,
)

# Set up Prometheus metrics
setup_metrics(app)

# Add logging context middleware (must be added before other middleware to capture context)
app.add_middleware(LoggingContextMiddleware)

# Add custom metrics middleware
app.add_middleware(MetricsMiddleware)

# Add profiling middleware (DEBUG only — saves HTML to /tmp on slow requests)
if settings.app.debug:
    app.add_middleware(ProfilingMiddleware)

# Add correlation ID middleware — must be outermost so request_id is set before all others
app.add_middleware(CorrelationIdMiddleware)

# Set up rate limiter exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# Unified error envelope: {error:{code,message,details}, meta:{request_id}}
# ---------------------------------------------------------------------------
_STATUS_CODE_NAMES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "CONFLICT",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMIT_EXCEEDED",
    500: "INTERNAL_ERROR",
}


def _error_response(status_code: int, code: str, message: str, details: list | None = None, headers=None) -> JSONResponse:
    """Build a uniform error response with the current request id."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {"code": code, "message": message, "details": details or []},
            "meta": {"request_id": correlation_id.get()},
        },
        headers=headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return validation errors in the unified envelope."""
    logger.warning("validation_error", path=request.url.path, errors=str(exc.errors()))
    details = [
        {
            "field": " -> ".join(str(part) for part in error["loc"] if part != "body"),
            "message": error["msg"],
        }
        for error in exc.errors()
    ]
    return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, "VALIDATION_ERROR", "Request validation failed", details)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Wrap HTTPExceptions (including auth headers) in the unified envelope."""
    code = _STATUS_CODE_NAMES.get(exc.status_code, "HTTP_ERROR")
    return _error_response(exc.status_code, code, str(exc.detail), headers=getattr(exc, "headers", None))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all: log the traceback, return a generic 500 (no internal details leaked)."""
    logger.exception("unhandled_exception", path=request.url.path)
    return _error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", "Internal server error")


# Set up CORS middleware.
# A wildcard origin is incompatible with credentialed requests (browsers reject
# `Access-Control-Allow-Origin: *` with credentials) and is unsafe in production.
_wildcard_cors = "*" in settings.app.allowed_origins
if _wildcard_cors and settings.app.environment == Environment.PRODUCTION:
    logger.warning("insecure_cors_wildcard_in_production", origins=settings.app.allowed_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.app.allowed_origins,
    allow_credentials=not _wildcard_cors,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.app.api_v1_str)


@app.get("/")
@limiter.limit(settings.rate_limit.endpoints["root"][0])
async def root(request: Request):
    """Root endpoint returning basic API information."""
    logger.info("root_endpoint_called")
    return {
        "name": settings.app.project_name,
        "version": settings.app.version,
        "status": "healthy",
        "environment": settings.app.environment.value,
        "swagger_url": "/docs",
        "redoc_url": "/redoc",
    }


@app.get("/health")
@limiter.limit(settings.rate_limit.endpoints["health"][0])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint with environment-specific information.

    Returns:
        JSONResponse: Health status payload, with HTTP 503 when the
        database is unreachable so load balancers can drop the instance.
    """
    logger.info("health_check_called")

    # Check database connectivity
    db_healthy = await database.health_check()

    response = {
        "status": "healthy" if db_healthy else "degraded",
        "version": settings.app.version,
        "environment": settings.app.environment.value,
        "components": {"api": "healthy", "database": "healthy" if db_healthy else "unhealthy"},
        "timestamp": datetime.now().isoformat(),
    }

    # If DB is unhealthy, set the appropriate status code
    status_code = status.HTTP_200_OK if db_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(content=response, status_code=status_code)
