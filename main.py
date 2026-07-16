#!/usr/bin/env python

"""Entry point for the FA-Crypto-Engine application."""

from config import AppConfig
from logger import AppLogger


def main() -> None:
    """Initialize the application and print the current configuration state."""
    config = AppConfig()
    logger = AppLogger("main")
    logger.info("FA-Crypto-Engine scaffold initialized.")
    print(f"Application configured for environment: {config.environment}")


if __name__ == "__main__":
    main()
