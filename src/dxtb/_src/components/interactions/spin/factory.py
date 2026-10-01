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
Spin Polarization: Factory
==========================

Factory function for the spin-polarization interaction.
"""

from __future__ import annotations

import torch

from dxtb._src.typing import Tensor
from dxtb._src.typing.exceptions import DeviceError

from .constants import MAX_SPIN_ELEMENT, load_spin_constants
from .spinpolarization import SpinPolarization

__all__ = ["new_spin_polarization"]


def new_spin_polarization(
    numbers: Tensor,
    wscale: Tensor | float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> SpinPolarization:
    """
    Create an instance of the spin-polarization interaction.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    wscale : Tensor | float, optional
        Scaling factor for all spin constants. Defaults to ``1.0``.
    device : torch.device | None, optional
        Device to store the tensor on. If ``None`` (default), the device is
        inferred from the `numbers` argument.
    dtype : torch.dtype | None, optional
        Data type of the tensor. If ``None`` (default), the default data type
        is used.

    Returns
    -------
    SpinPolarization
        Instance of the spin-polarization interaction.

    Raises
    ------
    ValueError
        No spin constants available for an element.
    DeviceError
        Passed device and device of `numbers` do not match.
    """
    if numbers.max() > MAX_SPIN_ELEMENT:
        raise ValueError(
            "Spin constants are only available up to atomic number "
            f"{MAX_SPIN_ELEMENT}."
        )

    if device is not None and device != numbers.device:
        raise DeviceError(
            f"Passed device ({device}) and device of `numbers` "
            f"({numbers.device}) do not match."
        )

    constants = load_spin_constants(device=numbers.device, dtype=dtype)
    return SpinPolarization(constants, wscale=wscale)
