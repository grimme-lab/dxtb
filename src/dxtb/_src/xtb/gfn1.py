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
xTB Hamiltonians: GFN1-xTB
==========================

The GFN1-xTB Hamiltonian.
"""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs

from dxtb import IndexHelper
from dxtb._src.components.interactions import Potential
from dxtb._src.param.base import Param
from dxtb._src.param.module import ParameterModule, ParamModule
from dxtb._src.typing import Any, Tensor, override

from .base import PAD, BaseHamiltonian

__all__ = ["GFN1Hamiltonian"]


class GFN1Hamiltonian(BaseHamiltonian):
    """Hamiltonian from GFN1-xTB parametrization."""

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
            from tad_mctc.ncoord import cn_d3

            self.cn = cn_d3

    def _get_hscale(self, par: ParamModule) -> Tensor:
        """
        Obtain the off-site scaling factor for the Hamiltonian.

        Returns
        -------
        Tensor
            Off-site scaling factor for the Hamiltonian.
        """
        if par.is_none("hamiltonian"):
            raise RuntimeError("No Hamiltonian specified.")

        # extract some vars for convenience
        kpol = par.get("hamiltonian.xtb.kpol")
        shell = par.get("hamiltonian.xtb.shell")
        ushells = self.ihelp.unique_angular

        angular2label = {
            0: "s",
            1: "p",
            2: "d",
            3: "f",
            4: "g",
        }
        angular_labels = [angular2label.get(int(ang), PAD) for ang in ushells]

        # precompute kii values outside loop (slightly faster)
        kii_values = []
        for i, label in enumerate(angular_labels):
            # For non-valence shells, use kpol
            if self.valence[i] == 0:
                kii_values.append(kpol)
                continue

            key = f"{label}{label}"
            if key in shell:
                # Use view(-1)[0] to keep the value as a tensor.
                val = shell[key]
                assert isinstance(val, ParameterModule)
                kii_values.append(val.param.view(-1)[0])
            else:
                kii_values.append(torch.tensor(1.0, **self.dd))

        n = len(ushells)
        ksh = torch.empty((n, n), **self.dd)
        for i in range(n):
            for j in range(i + 1):
                kii = kii_values[i]
                kjj = kii_values[j]
                if self.valence[i] == 1 and self.valence[j] == 1:
                    key1 = f"{angular_labels[i]}{angular_labels[j]}"
                    key2 = f"{angular_labels[j]}{angular_labels[i]}"

                    if key1 in shell:
                        val: ParameterModule = shell[key1]
                        ksh_val = val.param.view(-1)[0]
                    elif key2 in shell:
                        val: ParameterModule = shell[key2]
                        ksh_val = val.param.view(-1)[0]
                    else:
                        ksh_val = (kii + kjj) / 2.0

                else:
                    ksh_val = (kii + kjj) / 2.0

                # Assign symmetrically.
                ksh[i, j] = ksh_val
                ksh[j, i] = ksh_val

        # for i, ang_i in enumerate(ushells):
        #     ang_i = angular2label.get(int(ang_i.item()), PAD)

        #     if self.valence[i] == 0:
        #         kii = kpol
        #     else:
        #         kii = shell.get(f"{ang_i}{ang_i}", 1.0)

        #     for j, ang_j in enumerate(ushells):
        #         ang_j = angular2label.get(int(ang_j.item()), PAD)

        #         if self.valence[j] == 0:
        #             kjj = kpol
        #         else:
        #             kjj = shell.get(f"{ang_j}{ang_j}", 1.0)

        #         # only if both belong to the valence shell,
        #         # we will read from the parametrization
        #         if self.valence[i] == 1 and self.valence[j] == 1:
        #             # check both "sp" and "ps"
        #             ksh[i, j] = shell.get(
        #                 f"{ang_i}{ang_j}",
        #                 shell.get(
        #                     f"{ang_j}{ang_i}",
        #                     (kii + kjj) / 2.0,
        #                 ),
        #             )
        #         else:
        #             ksh[i, j] = (kii + kjj) / 2.0

        return ksh

    @override
    def _get_elem_valence(self, par: ParamModule) -> Tensor:
        """
        Obtain a mask for valence and non-valence shells. This is only required
        for GFN1-xTB's second hydrogen s-function.

        Parameters
        ----------
        par : ParamModule
            Representation of an extended tight-binding model.

        Returns
        -------
        Tensor
            Mask indicating valence shells for each unique species.
        """
        return par.get_elem_valence(self.unique, pad_val=PAD)

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
        """
        Calculate gradient of the full Hamiltonian with respect ot atomic
        positions.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        overlap : Tensor
            Overlap matrix.
        doverlap : Tensor
            Derivative of the overlap matrix.
        pmat : Tensor
            Density matrix.
        wmat : Tensor
            Energy-weighted density.
        pot : Tensor
            Self-consistent electrostatic potential.
        cn : Tensor
            Coordination number.

        Returns
        -------
        tuple[Tensor, Tensor]
            Derivative of energy with respect to coordination number (first
            tensor) and atomic positions (second tensor).
        """
        # masks
        mask_atom = real_pairs(self.numbers, mask_diagonal=False)
        mask_atom_diagonal = real_pairs(self.numbers, mask_diagonal=True)

        mask_shell = real_pairs(
            self.ihelp.spread_atom_to_shell(self.numbers), mask_diagonal=False
        )
        mask_shell_diagonal = self.ihelp.spread_atom_to_shell(
            mask_atom_diagonal, dim=(-2, -1)
        )

        mask_orb_diagonal = self.ihelp.spread_atom_to_orbital(
            mask_atom_diagonal, dim=(-2, -1)
        )

        zero = torch.tensor(0.0, **self.dd)

        # --------------------
        # Eq.28: X(EN_A, EN_B)
        # --------------------
        en = self.ihelp.spread_uspecies_to_shell(self.en)
        var_x = torch.where(
            mask_shell_diagonal,
            1.0
            + self.enscale
            * torch.pow(en.unsqueeze(-1) - en.unsqueeze(-2), 2.0),
            zero,
        )

        # --------------------
        # Eq.23: K_{AB}^{l,l'}
        # --------------------
        kpair = self.ihelp.spread_uspecies_to_shell(self.kpair, dim=(-2, -1))
        hscale = self.ihelp.spread_ushell_to_shell(self.hscale, dim=(-2, -1))
        valence = self.ihelp.spread_ushell_to_shell(self.valence)

        var_k = torch.where(
            valence.unsqueeze(-1) * valence.unsqueeze(-2),
            hscale * kpair * var_x,
            hscale,
        )

        # ----------------------
        # Eq.24: PI(R_AB, l, l')
        # ----------------------
        distances = storch.cdist(positions, positions, p=2)
        rad = self.ihelp.spread_uspecies_to_atom(self.rad)

        rr = storch.divide(distances, rad.unsqueeze(-1) + rad.unsqueeze(-2))
        rr_shell = self.ihelp.spread_atom_to_shell(
            torch.where(mask_atom_diagonal, storch.sqrt(rr), zero),
            (-2, -1),
        )

        shpoly = self.ihelp.spread_ushell_to_shell(self.shpoly)
        shpoly_a = shpoly.unsqueeze(-1)
        tmp_a = 1.0 + shpoly_a * rr_shell
        shpoly_b = shpoly.unsqueeze(-2)
        tmp_b = 1.0 + shpoly_b * rr_shell
        var_pi = tmp_a * tmp_b

        # ------------
        # Eq.23: H_EHT
        # ------------

        # `kcn` differs from paper (Eq.29) to be consistent with GFN2
        kcn = self.ihelp.spread_ushell_to_shell(self.kcn)
        selfenergy = self.ihelp.spread_ushell_to_shell(
            self.selfenergy
        ) - kcn * self.ihelp.spread_atom_to_shell(cn)

        var_h = torch.where(
            mask_shell,
            0.5 * (selfenergy.unsqueeze(-1) + selfenergy.unsqueeze(-2)),
            zero,
        )

        hcore = self.ihelp.spread_shell_to_orbital(
            torch.where(
                mask_shell_diagonal,
                var_pi * var_k * var_h,  # scale only off-diagonals
                var_h,
            ),
            dim=(-2, -1),
        )

        # ----------------------------------------------------------------------
        # Derivative of the electronic energy w.r.t. the atomic positions r
        # ----------------------------------------------------------------------
        # dE/dr = dE_EHT/dr + dE_coulomb/dr + dL_constraint/dr
        #       = [2*P*H - 2*W - P*(v + v^T)] * dS/dr + 2*P*H*S * dPI/dr / PI
        # ----------------------------------------------------------------------

        # ------------------------------------------------------------
        # derivative of Eq.24: PI(R_AB, l, l') -> dPI/dr (without rij)
        # ------------------------------------------------------------
        distances_shell = self.ihelp.spread_atom_to_shell(distances, (-2, -1))
        dvar_pi = torch.where(
            mask_shell_diagonal,
            (tmp_a * shpoly_b + tmp_b * shpoly_a)
            * rr_shell
            * 0.5
            / torch.pow(distances_shell, 2.0),
            zero,
        )

        # xTB Hamiltonian (without overlap, Hcore) times density matrix
        ph = pmat * hcore

        # E_EHT derivative for scaling function `PI` (2*P*H*S * dPI/dr / PI)
        dpi = (
            2
            * self.ihelp.reduce_orbital_to_shell(ph * overlap, dim=(-2, -1))
            * dvar_pi
            / var_pi
        )

        # factors for all derivatives of the overlap (2*P*H - 2*W - P*(V + V^T))
        tmp = 2 * (ph - wmat)
        if pot.mono is not None:
            tmp -= pmat * (pot.mono.unsqueeze(-1) + pot.mono.unsqueeze(-2))

        sval = torch.where(mask_orb_diagonal, tmp, zero)

        # distance vector from dR_AB/dr_A
        # (n_batch, atoms_i, atoms_j, 3)
        rij = torch.where(
            mask_atom.unsqueeze(-1),
            positions.unsqueeze(-2) - positions.unsqueeze(-3),
            zero,
        )

        # (n_batch, shells_i, shells_j) -> (n_batch, atoms_i, atoms_j)
        dpi = self.ihelp.reduce_shell_to_atom(dpi, dim=(-2, -1))

        # Multiplying with `rij` automatically includes the sign change on
        # switching atoms, which is manually done in the Fortran code.
        # (n_batch, atoms_i, atoms_j, 3) -> (n_batch, atoms_i, 3)
        g1 = torch.sum(dpi.unsqueeze(-1) * rij, dim=-2)

        # We cannot use the autograd of the overlap since the returned shape
        # will be (n_batch, atoms_i, 3). We need to multiply in an orbital-
        # resolved fashion before reducing.
        # (n_batch, orbs_i, orbs_j, 3) -> (n_batch, atoms_i, orbs_j, 3)
        ds = self.ihelp.reduce_orbital_to_atom(
            doverlap * sval.unsqueeze(-1), dim=-3, extra=True
        )

        # The Fortran code only calculates a triangular matrix and distributes
        # a positive gradient contribution to the ith atom and a negative
        # gradient contribution to the jth atom. Here, we have the full matrix,
        # which is why we get the same numeric value after summing along -2.
        # (n_batch, atoms_i, orbs_i, 3) -> (n_batch, atoms_i, 3)
        g2 = torch.sum(ds, dim=-2)

        # we cannot sum after adding both contributions (different shapes!)
        gradient = g1 + g2

        # ----------------------------------------------------------------------
        # Derivative of the electronic energy w.r.t. the coordination number
        # ----------------------------------------------------------------------
        # E = P * H = P * 0.5 * (H_mm(CN) + H_nn(CN)) * S * F
        # -> with: H_mm(CN) = se_mm(CN) = selfenergy - kcn * CN
        #          F = PI(R_AB, l, l') * K_{AB}^{l,l'} * X(EN_A, EN_B)
        # ----------------------------------------------------------------------

        # `kcn` differs from paper (Eq.29) to be consistent with GFN2
        dsedcn = -self.ihelp.spread_ushell_to_shell(self.kcn).unsqueeze(-2)

        # avoid symmetric matrix by only passing `dsedcn` vector, which must be
        # unsqueeze(-2)'d for batched calculations
        dhdcn = torch.where(
            mask_shell_diagonal,
            dsedcn * var_pi * var_k,  # only scale off-diagonals
            dsedcn,
        )

        # reduce orbital-resolved `P*S` for mult with shell-resolved `dhdcn`
        dcn = (
            self.ihelp.reduce_orbital_to_shell(pmat * overlap, dim=(-2, -1))
            * dhdcn
        )

        # reduce to atoms and sum for vector (requires non-symmetric matrix)
        dedcn = self.ihelp.reduce_shell_to_atom(dcn, dim=(-2, -1))

        return dedcn.sum(-2), gradient
