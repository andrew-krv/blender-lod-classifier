"""Logger factory for addon modules."""

from __future__ import annotations

import logging

_LOGGER_PREFIX = "lod_classifier"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger namespace for the addon."""

    logger = logging.getLogger(f"{_LOGGER_PREFIX}.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
