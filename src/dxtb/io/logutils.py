"""
Logging
"""

from __future__ import annotations

import logging

from ..constants import defaults

LOG_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


DEFAULT_LOG_CONFIG = {
    "level": LOG_LEVELS[defaults.LOG_LEVEL],
    "format": "%(asctime)s %(levelname)s %(name)s::%(funcName)s -> %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
}


def get_logging_config(**kwargs) -> dict:
    d = DEFAULT_LOG_CONFIG.copy()
    d.update(kwargs)
    return d
