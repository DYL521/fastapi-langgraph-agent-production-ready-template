"""This file contains the main application entry point."""

from contextlib import asynccontextmanager
from datetime import datetime

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
from agent.core.cache import cache_service
from agent.core.config import (
    Environment,
    settings,
)
from agent.core.langgraph.graph import LangGraphAgent
from agent.services.llm import LLMService
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

    try:
        await cache_service.initialize()
    except Exception as e:
        logger.exception("cache_initialization_failed", error=str(e))

    llm_service = LLMService()
    agent = LangGraphAgent(llm_service)
    try:
        await agent.create_graph()
        logger.info("graph_pre_warmed")
    except Exception as e:
        logger.exception("graph_pre_warm_failed", error=str(e))

    app.state.agent = agent

    try:
        await memory_service.initialize()
    except Exception as e:
        logger.exception("memory_service_pre_warm_failed", error=str(e))

    yield

    await agent.close()
    await cache_service.close()
    await database.dispose()
    logger.info("application_shutdown")


app = FastAPI(
    title=settings.app.project_name,
    version=settings.app.version,
    description=settings.app.description,
    openapi_url=f"{settings.app.api_v1_str}/openapi.json",
    lifespan=lifespan,
)

setup_metrics(app)

app.add_middleware(LoggingContextMiddleware)
app.add_middleware(MetricsMiddleware)

if settings.app.debug:
    app.add_middleware(ProfilingMiddleware)

app.add_middleware(CorrelationIdMiddleware)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # pyright: ignore[reportArgumentType]


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
    """Catch-all: log the traceback, return a generic 500."""
    logger.exception("unhandled_exception", path=request.url.path)
    return _error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", "Internal server error")


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
    """Health check endpoint with database connectivity status."""
    logger.info("health_check_called")

    db_healthy = await database.health_check()

    response = {
        "status": "healthy" if db_healthy else "degraded",
        "version": settings.app.version,
        "environment": settings.app.environment.value,
        "components": {"api": "healthy", "database": "healthy" if db_healthy else "unhealthy"},
        "timestamp": datetime.now().isoformat(),
    }

    status_code = status.HTTP_200_OK if db_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(content=response, status_code=status_code)
