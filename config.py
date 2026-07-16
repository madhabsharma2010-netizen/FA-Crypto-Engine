"""Application-wide configuration primitives."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class AppConfig:
    """Secure configuration container for runtime settings."""

    environment: str = os.getenv("FA_ENV", "development")
    binance_api_key: str | None = os.getenv("BINANCE_API_KEY")
    binance_api_secret: str | None = os.getenv("BINANCE_API_SECRET")

    def is_production(self) -> bool:
        """Return True when the app is running in production mode."""
        return self.environment.lower() == "production"
