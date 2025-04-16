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
from tad_mctc.convert import any_to_tensor

from dxtb._src.constants import xtb
from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import DD, Tensor, get_default_dtype

from .hal import Halogen

__all__ = ["new_halogen"]


def new_halogen(
    unique: Tensor,
    par: Param | ParamModule,
    cutoff: Tensor | float | int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Halogen | None:
    """
    Create new instance of Halogen class.

    Parameters
    ----------
    unique : Tensor
        Unique elements in the system (shape: ``(nunique,)``).
    par : Param | ParamModule
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
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }

    # compatibility with previous version based on `Param`
    if not isinstance(par, ParamModule):
        par = ParamModule(par, **dd)

    if "halogen" not in par or par.is_none("halogen"):
        return None

    if cutoff is None:
        cutoff = xtb.DEFAULT_XB_CUTOFF

    return Halogen(
        damp=par.get("halogen.classical.damping"),
        rscale=par.get("halogen.classical.rscale"),
        bond_strength=par.get_elem_param(unique, "xbond", pad_val=0),
        cutoff=any_to_tensor(cutoff, **dd),
        **dd,
    )
