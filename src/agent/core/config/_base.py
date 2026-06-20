"""Shared building blocks for the settings package.

Centralizes environment detection, the `.env` file resolution order, and the
common `SettingsConfigDict` so each domain settings class stays small.
"""

import os
from enum import Enum
from pathlib import Path

from pydantic_settings import SettingsConfigDict


class Environment(str, Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


def get_environment() -> Environment:
    """Resolve the current environment from the APP_ENV variable."""
    match os.getenv("APP_ENV", "development").lower():
        case "production" | "prod":
            return Environment.PRODUCTION
        case "staging" | "stage":
            return Environment.STAGING
        case "test":
            return Environment.TEST
        case _:
            return Environment.DEVELOPMENT


def _project_root() -> Path:
    """Find the repo root (the directory containing pyproject.toml)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def _env_files() -> tuple[str, ...]:
    """Env files in ascending priority (last wins), mirroring the legacy loader.

    Real process environment variables still take precedence over all of these.
    """
    root = _project_root()
    env = get_environment().value
    names = (".env", ".env.local", f".env.{env}", f".env.{env}.local")
    return tuple(str(root / name) for name in names)


ENV_FILES = _env_files()


def settings_config(**overrides) -> SettingsConfigDict:
    """Build a SettingsConfigDict sharing the env-file and parsing conventions."""
    return SettingsConfigDict(
        env_file=ENV_FILES,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),  # allow fields like `model`
        **overrides,
    )


def split_csv(value: object) -> object:
    """Parse a comma-separated env string into a list; pass lists through."""
    if value is None or isinstance(value, list):
        return value
    if isinstance(value, str):
        cleaned = value.strip().strip("\"'")
        if not cleaned:
            return []
        return [item.strip() for item in cleaned.split(",") if item.strip()]
    return value
