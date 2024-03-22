"""
Print versions.
"""

from __future__ import annotations

import platform

import torch

from ...__version__ import __version__


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
