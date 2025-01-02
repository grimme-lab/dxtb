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
xTB Hamiltonians: Base
======================

Base class for xTB Hamiltonians.
"""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs
from tad_mctc.convert import symmetrize
from tad_mctc.data.radii import ATOMIC as ATOMIC_RADII
from tad_mctc.units import EV2AU

from dxtb import IndexHelper
from dxtb._src.param import Param
from dxtb._src.typing import CNFunction, PathLike, Tensor, TensorLike

from .abc import HamiltonianABC

__all__ = ["BaseHamiltonian"]

PAD = -1


class BaseHamiltonian(HamiltonianABC, TensorLike):
    """
    Base class for GFN Hamiltonians.

    For the Hamiltonians, no integral driver is needed. Therefore, the
    signatures are different from the integrals over atomic orbitals. The most
    important difference is the `build` method, which does not require the
    driver anymore and only takes the positions (and the overlap integral).
    """

    numbers: Tensor
    """Atomic numbers of the atoms in the system."""
    unique: Tensor
    """Unique species of the system."""

    par: Param
    """Representation of parametrization of xtb model."""

    ihelp: IndexHelper
    """Helper class for indexing."""

    hscale: Tensor
    """Off-site scaling factor for the Hamiltonian."""
    kcn: Tensor
    """Coordination number dependent shift of the self energy."""
    kpair: Tensor
    """Element-pair-specific parameters for scaling the Hamiltonian."""
    refocc: Tensor
    """Reference occupation numbers."""
    selfenergy: Tensor
    """Self-energy of each species."""
    shpoly: Tensor
    """Polynomial parameters for the distant dependent scaling."""
    valence: Tensor
    """
    Whether the shell belongs to the valence shell.
    Only requried for GFN1-xTB (second s-function for H).
    """

    en: Tensor
    """Pauling electronegativity of each species."""
    rad: Tensor
    """Van-der-Waals radius of each species."""

    cn: CNFunction | None
    """Coordination number function."""

    __slots__ = [
        "numbers",
        "unique",
        "par",
        "ihelp",
        "hscale",
        "kcn",
        "kpair",
        "refocc",
        "selfenergy",
        "shpoly",
        "valence",
        "en",
        "rad",
    ]

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        ihelp: IndexHelper,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **_,
    ) -> None:
        super().__init__(device, dtype)

        # check device of input tensors
        if any(tensor.device != self.device for tensor in (numbers, ihelp)):
            raise ValueError("All input tensors must be on the same device")

        self.numbers = numbers
        self.unique = torch.unique(numbers)
        self.par = par
        self.ihelp = ihelp

        self.label = self.__class__.__name__
        self._matrix = None

        # Initialize Hamiltonian parameters

        if self.par.hamiltonian is None:
            raise RuntimeError("Parametrization does not specify Hamiltonian.")

        # atom-resolved parameters
        self.rad = ATOMIC_RADII.to(**self.dd)[self.unique]
        self.en = self._get_elem_param("en")

        # shell-resolved element parameters
        self.kcn = self._get_elem_param("kcn")
        self.selfenergy = self._get_elem_param("levels")
        self.shpoly = self._get_elem_param("shpoly")
        self.refocc = self._get_elem_param("refocc")
        self.valence = self._get_elem_valence()

        # shell-pair-resolved pair parameters
        self.hscale = self._get_hscale()
        self.kpair = self._get_pair_param(self.par.hamiltonian.xtb.kpair)

        # unit conversion
        self.selfenergy = self.selfenergy * EV2AU
        self.kcn = self.kcn * EV2AU
        # dtype should always be correct as it always uses self.dtype
        if any(
            tensor.dtype != self.dtype
            for tensor in (
                self.hscale,
                self.kcn,
                self.kpair,
                self.refocc,
                self.selfenergy,
                self.shpoly,
                self.en,
                self.rad,
            )
        ):  # pragma: no cover
            raise ValueError("All tensors must have same dtype")

        # device should always be correct as it always uses self.device
        if any(
            tensor.device != self.device
            for tensor in (
                self.hscale,
                self.kcn,
                self.kpair,
                self.refocc,
                self.selfenergy,
                self.shpoly,
                self.valence,
                self.en,
                self.rad,
            )
        ):  # pragma: no cover
            raise ValueError("All tensors must be on the same device")

    @property
    def matrix(self) -> Tensor | None:
        return self._matrix

    @matrix.setter
    def matrix(self, mat: Tensor) -> None:
        self._matrix = mat

    def clear(self) -> None:
        """
        Clear the integral matrix.
        """
        self._matrix = None

    @property
    def requires_grad(self) -> bool:
        if self._matrix is None:
            return False

        return self._matrix.requires_grad

    def _get_elem_param(self, key: str) -> Tensor:
        """
        Obtain element parameters for species.

        Parameters
        ----------
        key : str
            Name of the parameter to be retrieved.

        Returns
        -------
        Tensor
            Parameters for each species.
        """
        # pylint: disable=import-outside-toplevel
        from dxtb._src.param import get_elem_param

        return get_elem_param(
            self.unique, self.par.element, key, pad_val=PAD, **self.dd
        )

    def _get_elem_valence(self) -> Tensor:
        """
        Obtain a mask for valence and non-valence shells. This is only required for GFN1-xTB's second hydrogen s-function. For GFN2-xTB, this is a dummy method, i.e., the mask is always ``True``.

        Returns
        -------
        Tensor
            Mask indicating valence shells for each unique species.
        """
        return torch.ones(
            len(self.ihelp.unique_angular), device=self.device, dtype=torch.bool
        )

    def _get_pair_param(self, pair: dict[str, float]) -> Tensor:
        """
        Obtain element-pair-specific parameters for all species.

        Parameters
        ----------
        pair : dict[str, float]
            Pair parametrization.

        Returns
        -------
        Tensor
            Pair parameters for each species.
        """
        # pylint: disable=import-outside-toplevel
        from dxtb._src.param import get_pair_param

        return get_pair_param(self.unique.tolist(), pair, **self.dd)

    def get_occupation(self) -> Tensor:
        """
        Obtain the reference occupation numbers for each orbital.
        """

        refocc = self.ihelp.spread_ushell_to_orbital(self.refocc)
        orb_per_shell = self.ihelp.spread_shell_to_orbital(
            self.ihelp.orbitals_per_shell
        )

        return torch.where(
            orb_per_shell != 0,
            refocc / orb_per_shell,
            torch.tensor(0, **self.dd),
        )

    def to_pt(self, path: PathLike | None = None) -> None:
        """
        Save the integral matrix to a file.

        Parameters
        ----------
        path : PathLike | None
            Path to the file where the integral matrix should be saved. If
            ``None``, the matrix is saved to the default location.
        """
        if path is None:
            path = f"{self.label.casefold()}.pt"

        torch.save(self.matrix, path)

    def build(self, positions: Tensor, overlap: Tensor | None = None) -> Tensor:
        """
        Build the xTB Hamiltonian.

        Parameters
        ----------
        positions : Tensor
            Atomic positions of molecular structure.
        overlap : Tensor | None, optional
            Overlap matrix. If ``None``, the true xTB Hamiltonian is *not*
            built. Defaults to ``None``.

        Returns
        -------
        Tensor
            Hamiltonian (always symmetric).
        """
        if self.par.hamiltonian is None:
            raise RuntimeError("No Hamiltonian specified.")

        # masks
        mask_atom_diagonal = real_pairs(self.numbers, mask_diagonal=True)
        mask_shell = real_pairs(
            self.ihelp.spread_atom_to_shell(self.numbers), mask_diagonal=False
        )
        mask_shell_diagonal = self.ihelp.spread_atom_to_shell(
            mask_atom_diagonal, dim=(-2, -1)
        )

        zero = torch.tensor(0.0, **self.dd)

        # ----------------
        # Eq.29: H_(mu,mu)
        # ----------------
        if self.cn is None:
            cn = torch.zeros_like(self.numbers, **self.dd)
        else:
            cn = self.cn(self.numbers, positions)

        kcn = self.ihelp.spread_ushell_to_shell(self.kcn)

        # formula differs from paper to be consistent with GFN2 -> "kcn" adapted
        selfenergy = self.ihelp.spread_ushell_to_shell(
            self.selfenergy
        ) - kcn * self.ihelp.spread_atom_to_shell(cn)

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
        var_pi = (1.0 + shpoly.unsqueeze(-1) * rr_shell) * (
            1.0 + shpoly.unsqueeze(-2) * rr_shell
        )

        # --------------------
        # Eq.28: X(EN_A, EN_B)
        # --------------------
        en = self.ihelp.spread_uspecies_to_shell(self.en)
        var_x = torch.where(
            mask_shell_diagonal,
            1.0
            + self.par.hamiltonian.xtb.enscale
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

        # ------------
        # Eq.23: H_EHT
        # ------------
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

        if overlap is not None:
            hcore = hcore * overlap

        # force symmetry to avoid problems through numerical errors
        h0 = symmetrize(hcore)
        self.matrix = h0
        return h0
