# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Logging
"""

from __future__ import annotations

import logging

from ..constants import defaults

__all__ = ["LOG_LEVELS", "DEFAULT_LOG_CONFIG", "get_logging_config"]


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
