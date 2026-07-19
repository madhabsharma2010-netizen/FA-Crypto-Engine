"""Logging helpers for the FA-Crypto-Engine project."""

from __future__ import annotations

import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


class AppLogger:
    """Simple structured logger wrapper for application use."""

    def __init__(self, name: str, log_file: str | None = None) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

            if log_file:
                log_path = PROJECT_ROOT / "logs" / log_file
                log_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_path)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def info(self, message: str) -> None:
        """Log an informational message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)
