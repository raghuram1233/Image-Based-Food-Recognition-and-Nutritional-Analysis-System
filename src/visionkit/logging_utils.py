"""Centralised logging configuration for visionkit CLIs."""

from __future__ import annotations

import logging

_CONFIGURED = False


def configure_logging(verbose: bool = False) -> None:
    """Configure root logging once, with a concise, readable format.

    Args:
        verbose: When ``True`` the level is ``DEBUG``; otherwise ``INFO``.
    """
    global _CONFIGURED
    level = logging.DEBUG if verbose else logging.INFO
    if _CONFIGURED:
        logging.getLogger().setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
