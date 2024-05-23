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
Halogen Bond Correction: Factory
================================

A factory function to create instances of the :class:`dxtb.components.Halogen`
class.
"""
from __future__ import annotations

import torch

from dxtb._src.constants import xtb
from dxtb._src.param import Param, get_elem_param
from dxtb._src.typing import DD, Tensor, get_default_dtype

from .hal import Halogen

__all__ = ["new_halogen"]


def new_halogen(
    numbers: Tensor,
    par: Param,
    cutoff: Tensor = torch.tensor(xtb.DEFAULT_XB_CUTOFF),
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Halogen | None:
    """
    Create new instance of Halogen class.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    par : Param
        Representation of an extended tight-binding model.
    cutoff : Tensor
        Real space cutoff for halogen bonding interactions (default: 20.0).

    Returns
    -------
    Halogen | None
        Instance of the Halogen class or ``None`` if no halogen bond
        correction is used.

    Raises
    ------
    ValueError
        If parametrization does not contain a halogen bond correction.
    """

    if hasattr(par, "halogen") is False or par.halogen is None:
        return None

    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }

    damp = torch.tensor(par.halogen.classical.damping, **dd)
    rscale = torch.tensor(par.halogen.classical.rscale, **dd)

    unique = torch.unique(numbers)
    bond_strength = get_elem_param(unique, par.element, "xbond", pad_val=0, **dd)

    return Halogen(damp, rscale, bond_strength, cutoff=cutoff, **dd)
