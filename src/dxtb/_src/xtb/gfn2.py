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
xTB Hamiltonians: GFN2-xTB
==========================

The GFN2-xTB Hamiltonian.
"""

from __future__ import annotations

from functools import partial

import torch
from tad_mctc import storch

from dxtb import IndexHelper
from dxtb._src.components.interactions import Potential
from dxtb._src.param.base import Param
from dxtb._src.param.module import ParameterModule, ParamModule
from dxtb._src.typing import Any, Tensor

from .base import PAD, BaseHamiltonian

__all__ = ["GFN2Hamiltonian"]


class GFN2Hamiltonian(BaseHamiltonian):
    """
    The GFN2-xTB Hamiltonian.
    """

    def __init__(
        self,
        numbers: Tensor,
        par: Param | ParamModule,
        ihelp: IndexHelper,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(numbers, par, ihelp, device, dtype)

        # coordination number function
        if "cn" in kwargs:
            self.cn = kwargs.pop("cn")
        else:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.ncoord import cn_d3, gfn2_count

            self.cn = partial(cn_d3, counting_function=gfn2_count)

    def _get_hscale(self, par: ParamModule) -> Tensor:
        """
        Obtain the off-site scaling factor for the Hamiltonian.

        Parameters
        ----------
        par : ParamModule
            Representation of an extended tight-binding model.

        Returns
        -------
        Tensor
            Off-site scaling factor for the Hamiltonian.
        """
        if par.is_none("hamiltonian"):
            raise RuntimeError("No Hamiltonian specified.")

        # extract some vars for convenience
        shell = par.get("hamiltonian.xtb.shell")
        wexp = par.get("hamiltonian.xtb.wexp")
        ushells = self.ihelp.unique_angular

        angular2label = {
            0: "s",
            1: "p",
            2: "d",
            3: "f",
            4: "g",
        }
        angular_labels = [angular2label.get(int(ang), PAD) for ang in ushells]

        # ----------------------
        # Eq.37: Y(z^A_l, z^B_m)
        # ----------------------
        z = par.get_elem_param(self.unique, "slater", pad_val=PAD)
        zi = z.unsqueeze(-1)
        zj = z.unsqueeze(-2)
        zmat = storch.pow(
            2 * storch.divide(storch.sqrt(zi * zj), (zi + zj)), wexp
        )

        ksh = torch.ones((len(ushells), len(ushells)), **self.dd)
        for i, ang_i in enumerate(ushells):
            ang_i = angular_labels[i]

            for j, ang_j in enumerate(ushells):
                ang_j = angular_labels[j]

                key1 = f"{ang_i}{ang_j}"
                key2 = f"{ang_j}{ang_i}"

                # Since the parametrization only contains "sp" (not "ps"),
                # we need to check both.
                # For some reason, the parametrization does not contain "sp"
                # or "ps", although the value is calculated from "ss" and "pp",
                # and hence, always the same. The paper, however, specifically
                # mentions this.
                # tblite: xtb/gfn2.f90::new_gfn2_h0spec
                if key1 in shell:
                    val: ParameterModule = shell[key1]
                    kij = val.param.view(-1)[0]
                elif f"{ang_j}{ang_i}" in shell:
                    val: ParameterModule = shell[key2]
                    kij = val.param.view(-1)[0]
                else:
                    key_ii = f"{ang_i}{ang_i}"
                    key_jj = f"{ang_j}{ang_j}"

                    if PAD not in (ang_i, ang_j):
                        if key_ii not in shell:
                            raise KeyError(
                                f"GFN2 Core Hamiltonian: Missing '{ang_i}"
                                f"{ang_i}' in shell."
                            )
                        if key_jj not in shell:  # pragma: no cover
                            raise KeyError(
                                f"GFN2 Core Hamiltonian: Missing '{ang_j}"
                                f"{ang_j}' in shell."
                            )
                        val_ii: ParameterModule = shell[key_ii]
                        val_jj: ParameterModule = shell[key_jj]
                        kij = 0.5 * (
                            val_ii.param.view(-1)[0] + val_jj.param.view(-1)[0]
                        )
                    else:
                        kij = 1.0  # dummy for padding

                ksh[i, j] = kij * zmat[i, j]

        return ksh

    def get_gradient(
        self,
        positions: Tensor,
        overlap: Tensor,
        doverlap: Tensor,
        pmat: Tensor,
        wmat: Tensor,
        pot: Potential,
        cn: Tensor,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("GFN2 not implemented yet.")
