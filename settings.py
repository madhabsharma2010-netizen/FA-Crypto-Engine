"""Typed settings module for the engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    """Application settings with a simple object-oriented interface."""

    app_name: str = "FA-Crypto-Engine"
    version: str = "0.1.0"
    log_level: str = "INFO"
    timezone: str = "UTC"


DEFAULT_SETTINGS = Settings()
