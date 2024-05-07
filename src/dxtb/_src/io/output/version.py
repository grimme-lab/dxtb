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
Print versions.
"""

from __future__ import annotations

import platform

import torch

from dxtb.__version__ import __version__

__all__ = [
    "get_short_version",
    "get_python_version",
    "get_pytorch_version_short",
]


def get_short_version() -> str:
    pytorch_version = get_pytorch_version_short()
    python_version = get_python_version()

    return (
        f"* dxtb version {__version__} running with Python {python_version} "
        f"and PyTorch {pytorch_version}\n"
    )


def get_python_version() -> str:
    return platform.python_version()


def get_pytorch_version_short() -> str:
    config = torch.__config__.show().split(",")
    for info in config:
        if "TORCH_VERSION=" in info:
            return info.strip().split("=")[-1]

    raise RuntimeError("Version string not found in config.")
