"""
Factories
=========

Factory functions for integral drivers.
"""
from __future__ import annotations

import torch
from tad_mctc.typing import Any, Tensor

from ..basis import IndexHelper
from ..constants import labels
from ..param import Param
from .base import IntDriver
from .driver import (
    IntDriverLibcint,
    IntDriverPytorch,
    IntDriverPytorchLegacy,
    IntDriverPytorchNoAnalytical,
)
from .driver.libcint import OverlapLibcint
from .driver.pytorch import OverlapPytorch

__all__ = ["new_driver", "new_overlap"]


def new_driver(
    name: str,
    numbers: Tensor,
    par: Param,
    ihelp: IndexHelper,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IntDriver:
    if name == labels.INTDRIVER_LIBCINT:
        return new_driver_libcint(numbers, par, ihelp, device=device, dtype=dtype)

    if name == labels.INTDRIVER_PYTORCH:
        return new_driver_pytorch(numbers, par, ihelp, device=device, dtype=dtype)

    if name == labels.INTDRIVER_PYTORCH2:
        return new_driver_pytorch2(numbers, par, ihelp, device=device, dtype=dtype)

    if name == labels.INTDRIVER_LEGACY:
        return new_driver_legacy(numbers, par, ihelp, device=device, dtype=dtype)

    raise ValueError(f"Unknown integral driver '{name}'.")


def new_driver_libcint(
    numbers: Tensor,
    par: Param,
    ihelp: IndexHelper,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IntDriverLibcint:
    return IntDriverLibcint(numbers, par, ihelp, device=device, dtype=dtype)


def new_driver_pytorch(
    numbers: Tensor,
    par: Param,
    ihelp: IndexHelper,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IntDriverPytorch:
    return IntDriverPytorch(numbers, par, ihelp, device=device, dtype=dtype)


def new_driver_pytorch2(
    numbers: Tensor,
    par: Param,
    ihelp: IndexHelper,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IntDriverPytorchNoAnalytical:
    return IntDriverPytorchNoAnalytical(numbers, par, ihelp, device=device, dtype=dtype)


def new_driver_legacy(
    numbers: Tensor,
    par: Param,
    ihelp: IndexHelper,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IntDriverPytorchLegacy:
    return IntDriverPytorchLegacy(numbers, par, ihelp, device=device, dtype=dtype)


################################################################################


def new_overlap(
    driver: int = labels.INTDRIVER_LIBCINT,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> OverlapLibcint | OverlapPytorch:
    # Determine which overlap class to instantiate based on the type
    if driver == labels.INTDRIVER_LIBCINT:
        return new_overlap_libcint(device=device, dtype=dtype, **kwargs)

    if driver in (
        labels.INTDRIVER_PYTORCH,
        labels.INTDRIVER_PYTORCH2,
        labels.INTDRIVER_LEGACY,
    ):
        return new_overlap_pytorch(device=device, dtype=dtype, **kwargs)

    raise ValueError(f"Unknown integral driver '{driver}'.")


def new_overlap_libcint(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> OverlapLibcint:
    return OverlapLibcint(device=device, dtype=dtype, **kwargs)


def new_overlap_pytorch(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> OverlapPytorch:
    return OverlapPytorch(device=device, dtype=dtype, **kwargs)
