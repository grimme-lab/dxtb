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
Isotropic Electrostatics: Factory
================================

A factory function to create instances of the :class:`dxtb.components.IES`
class.
"""

from __future__ import annotations

import torch
from tad_mctc.data import radii

from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import DD, Tensor, get_default_dtype

from .ies import IES

__all__ = ["new_ies"]


def new_ies(
    numbers: Tensor,
    par: Param | ParamModule,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IES | None:
    """
    Create new instance of IES class.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the system (shape: ``(natom,)``).
    par : Param | ParamModule
        Representation of an extended tight-binding model.

    Returns
    -------
    IES | None
        An instance of the IES class if the tight-binding model supports isotropic electrostatics,
        otherwise None.

    Raises
    ------
    ValueError
        If the tight-binding model does not contain an isotropic electrostatics component.
    """
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }

    if not isinstance(par, ParamModule):
        par = ParamModule(par, **dd)

    if "eeq" not in par or par.is_none("eeq"):
        return None

    cn = par.get("eeq.cn")
    if cn != "erf":
        raise ValueError(
            "The GFN0 IES component only supports erf coordination numbers."
        )

    max_element = int(numbers.max().item()) if numbers.numel() > 0 else 0
    elements = torch.arange(
        max_element + 1, dtype=numbers.dtype, device=numbers.device
    )

    return IES(
        chi=par.get_elem_param(elements, "eeq_chi", pad_val=0),
        eeq_kcn=par.get_elem_param(elements, "eeq_kcn", pad_val=0),
        eta=par.get_elem_param(elements, "eeq_eta", pad_val=0),
        rad=par.get_elem_param(elements, "eeq_rad", pad_val=0),
        rcov=radii.COV_D3(**dd),
        cutoff=par.get("eeq.cutoff"),
        cn_max=par.get("eeq.cn_max"),
        cn_kcn=par.get("eeq.kcn"),
        **dd,
    )
