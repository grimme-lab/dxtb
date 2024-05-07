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
Dispersion: Factory
===================

Function for creating a new instance of a Dispersion.
"""

from __future__ import annotations

import warnings

import torch

from dxtb._src.param import Param
from dxtb._src.typing import DD, Tensor, get_default_dtype
from dxtb._src.typing.exceptions import ParameterWarning
from dxtb._src.utils import convert_float_tensor

from .base import Dispersion
from .d3 import DispersionD3
from .d4 import DispersionD4

__all__ = ["new_dispersion"]


def new_dispersion(
    numbers: Tensor,
    par: Param,
    charge: Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Dispersion | None:
    """
    Create new instance of the Dispersion class.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    par : Param
        Representation of an extended tight-binding model.

    Returns
    -------
    Dispersion | None
        Instance of the Dispersion class or ``None`` if no dispersion is used.

    Raises
    ------
    ValueError
        Parametrization does not contain a dispersion correction.
    ValueError
        D4 parametrization is requested but no charge given.
    """
    if hasattr(par, "dispersion") is False or par.dispersion is None:
        warnings.warn("No dispersion scheme found.", ParameterWarning)
        return None

    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }

    if par.dispersion.d3 is not None and par.dispersion.d4 is None:
        param = convert_float_tensor(
            {
                "a1": par.dispersion.d3.a1,
                "a2": par.dispersion.d3.a2,
                "s6": par.dispersion.d3.s6,
                "s8": par.dispersion.d3.s8,
                "s9": par.dispersion.d3.s9,
            },
            **dd,
        )
        return DispersionD3(numbers, param, device=device, dtype=dtype)

    if par.dispersion.d4 is not None and par.dispersion.d3 is None:
        if charge is None:
            raise ValueError("The total charge is required for DFT-D4.")

        param = convert_float_tensor(
            {
                "a1": par.dispersion.d4.a1,
                "a2": par.dispersion.d4.a2,
                "s6": par.dispersion.d4.s6,
                "s8": par.dispersion.d4.s8,
                "s9": par.dispersion.d4.s9,
                "s10": par.dispersion.d4.s10,
            },
            **dd,
        )
        return DispersionD4(numbers, param, charge, device=device, dtype=dtype)

    if par.dispersion.d3 is not None and par.dispersion.d4 is not None:
        raise ValueError("Parameters for both D3 and D4 found. Please decide.")

    return None
