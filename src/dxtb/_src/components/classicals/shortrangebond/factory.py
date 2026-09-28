# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
Short-range Bond Correction: Factory
=====================================

A factory function to create instances of the :class:`dxtb.components.ShortRangeBond`
class.
"""

from __future__ import annotations

import torch
from tad_mctc import ncoord
from tad_mctc.data import radii

from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import DD, Tensor, get_default_dtype

from .srb import ShortRangeBond

__all__ = ["new_srb"]


def new_srb(
    numbers: Tensor,
    par: Param | ParamModule,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ShortRangeBond | None:
    """
    Create new instance of ShortRangeBond class.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the system (shape: ``(natom,)``).
    par : Param | ParamModule
        Representation of an extended tight-binding model.

    Returns
    -------
    ShortRangeBond | None
        An instance of the ShortRangeBond class if the tight-binding model supports short-range bond correction,
        otherwise None.

    Raises
    ------
    ValueError
        If he parametrization does not contain a short-range bond correction component.
    """
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }
    if not isinstance(par, ParamModule):
        par = ParamModule(par, **dd)

    if "short_range" not in par or par.is_none("short_range.srb"):
        return None

    cn = par.get("short_range.srb.cn")
    counting_function = getattr(ncoord, f"{cn}_count")

    return ShortRangeBond(
        r0=par.get_elem_param(torch.unique(numbers), "srb_r0", pad_val=0),
        cnfak=par.get_elem_param(torch.unique(numbers), "srb_cnfak", pad_val=0),
        en=par.get_elem_param(torch.unique(numbers), "srb_en", pad_val=0),
        rcov=radii.COV_D3(**dd),
        counting_function=counting_function,
        shift=par.get("short_range.srb.shift"),
        prefactor=par.get("short_range.srb.prefactor"),
        steepness=par.get("short_range.srb.steepness"),
        enscale=par.get("short_range.srb.enscale"),
        enpoly=par.get("short_range.srb.enpoly"),
        pair_cutoff2=par.get("short_range.srb.pair_cutoff2"),
        cn_cutoff=par.get("short_range.srb.cn_cutoff"),
        cn_max=par.get("short_range.srb.cn_max"),
        cn_kcn=par.get("short_range.srb.cn_kcn"),
        **dd,
    )
