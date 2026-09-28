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
xTB Hamiltonians: GFN0-xTB
==========================

The GFN0-xTB Hamiltonian.
"""

from __future__ import annotations

from functools import partial

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs
from tad_mctc.convert import any_to_tensor, symmetrize
from tad_mctc.data import radii
from tad_mctc.ncoord import coordination_number, erf_count
from tad_mctc.units import EV2AU
from tad_multicharge.model.eeq import EEQModel

from dxtb import IndexHelper
from dxtb._src.components.interactions import Potential
from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import Any, Self, Tensor, override

from .base import PAD, BaseHamiltonian

__all__ = ["GFN0Hamiltonian"]


class GFN0Hamiltonian(BaseHamiltonian):
    """Charge- and coordination-dependent GFN0-xTB Hamiltonian."""

    def __init__(
        self,
        numbers: Tensor,
        par: Param | ParamModule,
        ihelp: IndexHelper,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **_: Any,
    ) -> None:
        if not isinstance(par, ParamModule):
            par = ParamModule(par, device=device, dtype=dtype)

        super().__init__(numbers, par, ihelp, device, dtype)

        if par.is_none("eeq"):
            raise RuntimeError("GFN0 Hamiltonian requires EEQ parameters.")
        if par.get("eeq.cn") != "erf":
            raise ValueError("GFN0 Hamiltonian only supports erf CN.")

        # The standalone reference keeps H0 in eV and converts its electronic
        # energy afterwards. dxtb keeps H0 in Hartree, so convert every
        # energy-like H0 table with tad-mctc's canonical conversion here.
        self.selfenergy = (
            par.get_elem_param(self.unique, "levels", pad_val=0) * EV2AU
        )
        self.kcn = par.get_elem_param(self.unique, "kcn", pad_val=0) * EV2AU
        self.kq = par.get_elem_param(self.unique, "kq", pad_val=0) * EV2AU
        self.kqat = par.get_elem_param(self.unique, "kqat", pad_val=0) * EV2AU

        self.h0rad = par.get_elem_param(self.unique, "h0rad", pad_val=1)
        self.kdiff = par.get("hamiltonian.xtb.kdiff")
        self.enshell = par.get("hamiltonian.xtb.enshell")
        self.enscale4 = par.get("hamiltonian.xtb.enscale4")

        max_element = int(numbers.max().item()) if numbers.numel() > 0 else 0
        elements = torch.arange(
            max_element + 1, dtype=numbers.dtype, device=numbers.device
        )
        self.eeq_model = EEQModel(
            par.get_elem_param(elements, "eeq_chi", pad_val=0),
            par.get_elem_param(elements, "eeq_kcn", pad_val=0),
            par.get_elem_param(elements, "eeq_eta", pad_val=0),
            par.get_elem_param(elements, "eeq_rad", pad_val=0),
            **self.dd,
        )

        self.cn_radii = radii.COV_D3(**self.dd)[numbers]
        self.cn_cutoff = par.get("eeq.cutoff")
        self.cn_max = par.get("eeq.cn_max")
        self.cn_kcn = par.get("eeq.kcn")
        self._set_cn_callable()

    def _set_cn_callable(self) -> None:
        """Configure the local GFN0 coordination-number callable."""
        self.cn = partial(
            coordination_number,
            counting_function=erf_count,
            rcov=self.cn_radii,
            cutoff=self.cn_cutoff,
            cn_max=self.cn_max,
            kcn=self.cn_kcn,
        )

    @override
    def _clone_tensorlike(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Clone GFN0-owned tensors and its EEQ model with the base fields."""
        target_device = self.device if device is None else device
        target_dtype = self.dtype if dtype is None else dtype
        new = super()._clone_tensorlike(device=device, dtype=dtype)

        # TensorLike treats every plain tensor as floating-point. Restore the
        # integer and boolean indexing tensors with only a device conversion.
        new.numbers = self.numbers.to(device=target_device)
        new.unique = self.unique.to(device=target_device)
        new.valence = self.valence.to(device=target_device)
        new.ihelp = self.ihelp.to(device=target_device)
        if self.matrix is not None:
            new.matrix = self.matrix.to(
                device=target_device, dtype=target_dtype
            )

        owned_tensors = (
            "kq",
            "kqat",
            "h0rad",
            "kdiff",
            "enshell",
            "enscale4",
            "cn_radii",
            "cn_cutoff",
            "cn_max",
            "cn_kcn",
        )
        for name in owned_tensors:
            value = getattr(self, name)
            setattr(
                new,
                name,
                value.to(device=target_device, dtype=target_dtype),
            )

        model = self.eeq_model
        new.eeq_model = EEQModel(
            model.chi.to(device=target_device, dtype=target_dtype),
            model.kcn.to(device=target_device, dtype=target_dtype),
            model.eta.to(device=target_device, dtype=target_dtype),
            model.rad.to(device=target_device, dtype=target_dtype),
            device=target_device,
            dtype=target_dtype,
        )
        new._set_cn_callable()
        return new

    @override
    def _get_elem_valence(self, par: ParamModule) -> Tensor:
        """Return the GFN0 valence-shell mask, including duplicate H shells."""
        return par.get_elem_valence(self.unique, pad_val=PAD)

    @override
    def _get_hscale(self, par: ParamModule) -> Tensor:
        """Build shell-pair kScale multiplied by Slater-exponent weighting."""
        if par.is_none("hamiltonian"):
            raise RuntimeError("No Hamiltonian specified.")

        shell = par.get("hamiltonian.xtb.shell", unwrapped=False)
        wexp = par.get("hamiltonian.xtb.wexp")
        angular = self.ihelp.unique_angular
        labels = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}
        angular_labels = [labels.get(int(ang), PAD) for ang in angular]

        zeta = par.get_elem_param(self.unique, "slater", pad_val=1)
        zi = zeta.unsqueeze(-1)
        zj = zeta.unsqueeze(-2)
        zeta_weight = storch.pow(
            2.0 * storch.divide(storch.sqrt(zi * zj), zi + zj), wexp
        )

        kscale = torch.ones((len(angular), len(angular)), **self.dd)
        for i, label_i in enumerate(angular_labels):
            for j, label_j in enumerate(angular_labels):
                key_ij = f"{label_i}{label_j}"
                key_ji = f"{label_j}{label_i}"
                if key_ij in shell:
                    value = par.get(f"hamiltonian.xtb.shell.{key_ij}")
                elif key_ji in shell:
                    value = par.get(f"hamiltonian.xtb.shell.{key_ji}")
                elif PAD in (label_i, label_j):
                    value = torch.tensor(1.0, **self.dd)
                else:  # pragma: no cover - validated built-in data is complete
                    raise KeyError(
                        f"GFN0 Hamiltonian: missing shell pair '{key_ij}'."
                    )
                kscale[i, j] = value

        return kscale * zeta_weight

    def get_coordination_number(self, positions: Tensor) -> Tensor:
        """Evaluate capped GFN0 coordination numbers."""
        if self.cn is None:  # pragma: no cover - initialized above
            raise RuntimeError("GFN0 coordination-number function is missing.")
        return self.cn(self.numbers, positions)

    def get_eeq_charges(
        self, positions: Tensor, charge: Tensor | float | int, cn: Tensor
    ) -> Tensor:
        """Solve the coordinate-local main GFN0 EEQ model."""
        total_charge = any_to_tensor(charge, **self.dd)
        charges = self.eeq_model.solve(
            self.numbers, positions, total_charge, cn
        )
        assert isinstance(charges, Tensor)
        return charges

    def get_selfenergy(self, cn: Tensor, charges: Tensor) -> Tensor:
        """Return environment-shifted self-energies for every atom shell."""
        eps0 = self.ihelp.spread_ushell_to_shell(self.selfenergy)
        kcn = self.ihelp.spread_ushell_to_shell(self.kcn)
        kq = self.ihelp.spread_ushell_to_shell(self.kq)
        shell_cn = self.ihelp.spread_atom_to_shell(cn)
        shell_q = self.ihelp.spread_atom_to_shell(charges)

        kqat = self.ihelp.spread_uspecies_to_atom(self.kqat)
        shell_q2 = self.ihelp.spread_atom_to_shell(kqat * charges**2)
        return eps0 - kcn * shell_cn - kq * shell_q - shell_q2

    @override
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
        raise NotImplementedError(
            "GFN0 analytical gradient is not implemented."
        )

    @override
    def build(
        self,
        positions: Tensor,
        overlap: Tensor | None = None,
        charge: Tensor | float | int | None = None,
    ) -> Tensor:
        """Build the GFN0 H0 matrix from local CN and main EEQ charges."""
        if charge is None:
            raise ValueError("Total molecular charge is required for GFN0 H0.")

        zero = torch.tensor(0.0, **self.dd)
        atom_pairs = real_pairs(self.numbers, mask_diagonal=True)
        shell_pairs = self.ihelp.spread_atom_to_shell(atom_pairs, dim=(-2, -1))

        cn = self.get_coordination_number(positions)
        charges = self.get_eeq_charges(positions, charge, cn)
        selfenergy = self.get_selfenergy(cn, charges)

        # Exact GFN0 distance-dependent shell polynomial. The stored shpoly
        # values already contain the reference function's factor of 0.01.
        distances = storch.cdist(positions, positions, p=2)
        h0rad = self.ihelp.spread_uspecies_to_atom(self.h0rad)
        reduced = storch.divide(
            distances, h0rad.unsqueeze(-1) + h0rad.unsqueeze(-2)
        )
        root_reduced = self.ihelp.spread_atom_to_shell(
            torch.where(atom_pairs, storch.sqrt(reduced), zero),
            dim=(-2, -1),
        )
        shpoly = self.ihelp.spread_ushell_to_shell(self.shpoly)
        distance_scale = (1.0 + shpoly.unsqueeze(-1) * root_reduced) * (
            1.0 + shpoly.unsqueeze(-2) * root_reduced
        )

        # Shell-dependent quadratic and quartic electronegativity polynomial.
        angular = self.ihelp.unique_angular.clamp(min=0)
        shell_en = self.enshell[angular]
        enscale = 0.005 * (shell_en.unsqueeze(-1) + shell_en.unsqueeze(-2))
        enscale = self.ihelp.spread_ushell_to_shell(enscale, dim=(-2, -1))
        en = self.ihelp.spread_uspecies_to_shell(self.en)
        den2 = (en.unsqueeze(-1) - en.unsqueeze(-2)) ** 2
        enpoly = 1.0 + enscale * den2 + self.enscale4 * enscale * den2**2

        kpair = self.ihelp.spread_uspecies_to_shell(self.kpair, dim=(-2, -1))
        hscale = self.ihelp.spread_ushell_to_shell(self.hscale, dim=(-2, -1))
        scale = hscale * kpair * enpoly

        valence = self.ihelp.spread_ushell_to_shell(self.valence)
        both_valence = valence.unsqueeze(-1) & valence.unsqueeze(-2)
        one_nonvalence = valence.unsqueeze(-1) ^ valence.unsqueeze(-2)
        scale = torch.where(
            both_valence,
            scale,
            torch.where(one_nonvalence, scale * self.kdiff, zero),
        )

        average = 0.5 * (selfenergy.unsqueeze(-1) + selfenergy.unsqueeze(-2))
        shell_h0 = torch.where(
            shell_pairs, distance_scale * scale * average, zero
        )
        h0 = self.ihelp.spread_shell_to_orbital(shell_h0, dim=(-2, -1))
        if overlap is not None:
            h0 = h0 * overlap

        # The reference sets only true orbital diagonals on-site. In
        # particular, the duplicate hydrogen s shells have no on-site block.
        diagonal = self.ihelp.spread_shell_to_orbital(selfenergy)
        h0 = symmetrize(h0) + torch.diag_embed(diagonal)
        self.matrix = h0
        return h0
