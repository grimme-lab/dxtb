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
from dxtb._src.param import Param
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
        par: Param,
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
            from tad_mctc.ncoord import cn_d3, gfn2_count

            self.cn = partial(cn_d3, counting_function=gfn2_count)

    def _get_hscale(self) -> Tensor:
        """
        Obtain the off-site scaling factor for the Hamiltonian.

        Returns
        -------
        Tensor
            Off-site scaling factor for the Hamiltonian.
        """
        if self.par.hamiltonian is None:
            raise RuntimeError("No Hamiltonian specified.")

        # extract some vars for convenience
        shell = self.par.hamiltonian.xtb.shell
        wexp = self.par.hamiltonian.xtb.wexp
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
        z = self._get_elem_param("slater")
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

                # Since the parametrization only contains "sp" (not "ps"),
                # we need to check both.
                # For some reason, the parametrization does not contain "sp"
                # or "ps", although the value is calculated from "ss" and "pp",
                # and hence, always the same. The paper, however, specifically
                # mentions this.
                # tblite: xtb/gfn2.f90::new_gfn2_h0spec
                if f"{ang_i}{ang_j}" in shell:
                    kij = shell[f"{ang_i}{ang_j}"]
                elif f"{ang_j}{ang_i}" in shell:
                    kij = shell[f"{ang_j}{ang_i}"]
                else:
                    if ang_i != PAD and ang_j != PAD:
                        if f"{ang_i}{ang_i}" not in shell:
                            raise KeyError(
                                f"GFN2 Core Hamiltonian: Missing '{ang_i}"
                                f"{ang_i}' in shell."
                            )
                        if f"{ang_j}{ang_j}" not in shell:  # pragma: no cover
                            raise KeyError(
                                f"GFN2 Core Hamiltonian: Missing '{ang_j}"
                                f"{ang_j}' in shell."
                            )

                        kij = 0.5 * (
                            shell[f"{ang_i}{ang_i}"] + shell[f"{ang_j}{ang_j}"]
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
