"""Dialect-aware SQLAlchemy URL construction for the relational database.

Used by the sync Database engine and by Alembic so neither hardcodes a
``postgresql://`` prefix. Supports PostgreSQL (default) and MySQL 8+.
"""

from urllib.parse import quote_plus

from agent.core.config import settings

# (dialect, async) -> SQLAlchemy driver prefix
_DRIVERS = {
    ("postgres", False): "postgresql+psycopg2",
    ("postgres", True): "postgresql+psycopg",
    ("mysql", False): "mysql+pymysql",
    ("mysql", True): "mysql+asyncmy",
}


def build_sqlalchemy_url(async_: bool = False) -> str:
    """Build a SQLAlchemy connection URL for the configured DB dialect.

    Args:
        async_: When True, return an async driver URL (psycopg / asyncmy);
            otherwise a sync driver URL (psycopg2 / pymysql).

    Returns:
        A SQLAlchemy URL string, e.g. ``postgresql+psycopg2://user:pw@host:5432/db``.

    Raises:
        ValueError: If ``settings.database.dialect`` is not supported.
    """
    driver = _DRIVERS.get((settings.database.dialect, async_))
    if driver is None:
        raise ValueError(f"unsupported DB_DIALECT: {settings.database.dialect!r} (expected 'postgres' or 'mysql')")

    user = quote_plus(settings.database.user)
    password = quote_plus(settings.database.password)
    url = f"{driver}://{user}:{password}@{settings.database.host}:{settings.database.port}/{settings.database.name}"

    # MySQL connections default to latin1; force utf8mb4 so the connection
    # collation matches the utf8mb4 columns (avoids "illegal mix of collations").
    if settings.database.dialect == "mysql":
        url += "?charset=utf8mb4"
    return url
