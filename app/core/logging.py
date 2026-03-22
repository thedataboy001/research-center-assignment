# core/logging.py
# Structured logging configuration.
# Using Python's built-in logging with a consistent format so that
# log lines can be parsed by log aggregation tools (e.g. CloudWatch, Datadog).

import logging
import sys
from app.core.config import settings


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance.

    Usage:
        from app.core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Training started")

    Args:
        name: typically __name__ of the calling module

    Returns:
        Configured Logger instance 
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    # Set log level based on environment
    level = logging.DEBUG if settings.debug else logging.INFO
    logger.setLevel(level)

    # Console handler — outputs to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Structured format: timestamp | level | module | message
    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | "
            "%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger