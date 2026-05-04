"""
Logging configuration for the research evaluation and diagnostic pipeline.
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_file: Path | None = None):
    """
    Configure the global Loguru logger for the pipeline.

    Sets up a stderr sink for user-facing info and an optional file sink for deep debugging.
    Third-party noise from libraries like 'httpx' is automatically suppressed.

    Args:
        log_file: Path to a file where detailed debug logs should be persisted.
    """
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    if log_file:
        logger.add(log_file, level="DEBUG", rotation="10 MB")
