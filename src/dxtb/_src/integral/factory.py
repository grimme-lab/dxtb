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
Factories
=========

Factory functions for integral drivers.
"""

from __future__ import annotations

import torch

from dxtb import IndexHelper
from dxtb._src.constants import labels
from dxtb._src.param import Param
from dxtb._src.typing import TYPE_CHECKING, Any, Tensor

from .base import IntDriver

if TYPE_CHECKING:
    from .driver.libcint import IntDriverLibcint, OverlapLibcint
    from .driver.pytorch import (
        IntDriverPytorch,
        IntDriverPytorchLegacy,
        IntDriverPytorchNoAnalytical,
        OverlapPytorch,
    )

__all__ = ["new_driver", "new_overlap"]


def new_driver(
    name: int,
    numbers: Tensor,
    par: Param,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IntDriver:
    if name == labels.INTDRIVER_LIBCINT:
        return new_driver_libcint(numbers, par, device=device, dtype=dtype)

    if name == labels.INTDRIVER_ANALYTICAL:
        return new_driver_pytorch(numbers, par, device=device, dtype=dtype)

    if name == labels.INTDRIVER_AUTOGRAD:
        return new_driver_pytorch2(numbers, par, device=device, dtype=dtype)

    if name == labels.INTDRIVER_LEGACY:
        return new_driver_legacy(numbers, par, device=device, dtype=dtype)

    raise ValueError(f"Unknown integral driver '{ labels.INTDRIVER_MAP[name]}'.")


def new_driver_libcint(
    numbers: Tensor,
    par: Param,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IntDriverLibcint:
    # pylint: disable=import-outside-toplevel
    from .driver.libcint import IntDriverLibcint as IntDriver

    ihelp = IndexHelper.from_numbers(numbers, par)
    return IntDriver(numbers, par, ihelp, device=device, dtype=dtype)


def new_driver_pytorch(
    numbers: Tensor,
    par: Param,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IntDriverPytorch:
    # pylint: disable=import-outside-toplevel
    from .driver.pytorch import IntDriverPytorch as IntDriver

    ihelp = IndexHelper.from_numbers(numbers, par)
    return IntDriver(numbers, par, ihelp, device=device, dtype=dtype)


def new_driver_pytorch2(
    numbers: Tensor,
    par: Param,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IntDriverPytorchNoAnalytical:
    # pylint: disable=import-outside-toplevel
    from .driver.pytorch import IntDriverPytorchNoAnalytical as IntDriver

    ihelp = IndexHelper.from_numbers(numbers, par)
    return IntDriver(numbers, par, ihelp, device=device, dtype=dtype)


def new_driver_legacy(
    numbers: Tensor,
    par: Param,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IntDriverPytorchLegacy:
    # pylint: disable=import-outside-toplevel
    from .driver.pytorch import IntDriverPytorchLegacy as IntDriver

    ihelp = IndexHelper.from_numbers(numbers, par)
    return IntDriver(numbers, par, ihelp, device=device, dtype=dtype)


################################################################################


def new_overlap(
    driver: int = labels.INTDRIVER_LIBCINT,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> OverlapLibcint | OverlapPytorch:
    # Determine which overlap class to instantiate based on the type
    if driver == labels.INTDRIVER_LIBCINT:
        if kwargs.pop("force_cpu_for_libcint", True):
            device = torch.device("cpu")
        return new_overlap_libcint(device=device, dtype=dtype, **kwargs)

    if driver in (
        labels.INTDRIVER_ANALYTICAL,
        labels.INTDRIVER_AUTOGRAD,
        labels.INTDRIVER_LEGACY,
    ):
        return new_overlap_pytorch(device=device, dtype=dtype, **kwargs)

    raise ValueError(f"Unknown integral driver '{driver}'.")


def new_overlap_libcint(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> OverlapLibcint:
    # pylint: disable=import-outside-toplevel
    from .driver.libcint import OverlapLibcint

    return OverlapLibcint(device=device, dtype=dtype, **kwargs)


def new_overlap_pytorch(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> OverlapPytorch:
    # pylint: disable=import-outside-toplevel
    from .driver.pytorch import OverlapPytorch

    return OverlapPytorch(device=device, dtype=dtype, **kwargs)
