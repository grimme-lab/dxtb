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
Integral driver: Factories
==========================

Factory functions for integral drivers.
"""

from __future__ import annotations

import torch

from dxtb import IndexHelper
from dxtb._src.constants import labels
from dxtb._src.param import Param
from dxtb._src.typing import TYPE_CHECKING, Tensor

from ..base import IntDriver

if TYPE_CHECKING:
    from .libcint import IntDriverLibcint
    from .pytorch import (
        IntDriverPytorch,
        IntDriverPytorchLegacy,
        IntDriverPytorchNoAnalytical,
    )

__all__ = ["new_driver"]


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
        return new_driver_pytorch_no_analytical(
            numbers, par, device=device, dtype=dtype
        )

    if name == labels.INTDRIVER_LEGACY:
        return new_driver_legacy(numbers, par, device=device, dtype=dtype)

    raise ValueError(f"Unknown integral driver '{labels.INTDRIVER_MAP[name]}'.")


################################################################################


def new_driver_libcint(
    numbers: Tensor,
    par: Param,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IntDriverLibcint:
    # pylint: disable=import-outside-toplevel
    from .libcint import IntDriverLibcint as _IntDriver

    ihelp = IndexHelper.from_numbers(numbers, par)
    return _IntDriver(numbers, par, ihelp, device=device, dtype=dtype)


################################################################################


def new_driver_pytorch(
    numbers: Tensor,
    par: Param,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IntDriverPytorch:
    # pylint: disable=import-outside-toplevel
    from .pytorch import IntDriverPytorch as _IntDriver

    ihelp = IndexHelper.from_numbers(numbers, par)
    return _IntDriver(numbers, par, ihelp, device=device, dtype=dtype)


def new_driver_pytorch_no_analytical(
    numbers: Tensor,
    par: Param,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IntDriverPytorchNoAnalytical:
    # pylint: disable=import-outside-toplevel
    from .pytorch import IntDriverPytorchNoAnalytical as _IntDriver

    ihelp = IndexHelper.from_numbers(numbers, par)
    return _IntDriver(numbers, par, ihelp, device=device, dtype=dtype)


def new_driver_legacy(
    numbers: Tensor,
    par: Param,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IntDriverPytorchLegacy:
    # pylint: disable=import-outside-toplevel
    from .pytorch import IntDriverPytorchLegacy as _IntDriver

    ihelp = IndexHelper.from_numbers(numbers, par)
    return _IntDriver(numbers, par, ihelp, device=device, dtype=dtype)
