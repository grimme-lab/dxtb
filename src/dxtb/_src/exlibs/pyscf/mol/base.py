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
# pylint: disable=protected-access
"""
PySCF: Moleclue
===============

This module contains a class for a molecular representation in PySCF's format.
It also contains the short-cut version as in PySCF (``M``).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from pyscf import gto  # type: ignore
from tad_mctc.convert import tensor_to_numpy
from tad_mctc.data import pse
from tad_mctc.molecule.container import Mol

from dxtb._src.typing import Tensor

__all__ = ["PyscfMol", "M"]


# Turn off PySCF's normalization since dxtb's normalization is different,
# requiring a separate normalization anyway.
gto.mole.NORMALIZE_GTO = False


def M(
    numbers: Tensor, positions: Tensor, xtb_version: str = "gfn1", **kwargs
) -> PyscfMol:
    """
    Shortcut for building up the `PyscfMol`.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    xtb_version : str, optional
        Version of xtb to use for basis set. Defaults to "gfn1".

    Returns
    -------
    PyscfMol
        Built-up PySCF molecule.
    """
    mol = PyscfMol(numbers, positions, xtb_version=xtb_version)
    mol.build(**kwargs)
    mol.undo_norm_prim()
    return mol


class PyscfMol(gto.Mole):
    """
    Pyscf's molecule representation that can be created upon passing only
    ``numbers`` and ``positions``. Note that the basis set is created from a
    database and only the xtb version must be provided during instantiation.

    .. warning::

        You still have to initialize the molecule via `mol.build()`.
    """

    def __init__(
        self,
        numbers: Tensor,
        positions: Tensor,
        xtb_version: str = "gfn1",
        unit: str = "B",
        **kwargs,
    ):
        # change default unit to Bohr (a.u.)
        if "unit" in kwargs:
            warnings.warn(
                "Unit in kwargs will be overwritten. Please change the unit "
                "using the argument in the constructor of the class.",
                UserWarning,
            )
        kwargs["unit"] = unit

        # init pyscf's molecule type
        super().__init__(**kwargs)

        # path to basis set storage
        path = Path(Path(__file__).resolve().parent, "basis", xtb_version)
        if not path.exists():
            raise FileNotFoundError(f"Path '{path}' for basis does not exist.")

        # internal format of `pyscf.gto.Mole.atom`
        # atom = [[atom1, (x, y, z)],
        #         [atom2, (x, y, z)],
        #         ...
        #         [atomN, (x, y, z)]]
        atom: list[list[str | np.ndarray]] = []

        basis: dict[str, list] = {}
        for i, number in enumerate(numbers.tolist()):
            if number not in pse.Z2S:
                raise ValueError(f"Atom '{number}' not found.")

            symbol = pse.Z2S[number]
            pos = tensor_to_numpy(positions[i, :])
            atom.append([symbol, pos])

            # pyscf expects nwchem format of basis set
            p = Path(path, f"{number:02d}.nwchem")
            with open(p, encoding="utf8") as f:
                content = f.read()
                basis[f"{pse.Z2S[number]}"] = gto.parse(content)

        self.atom = atom
        self.basis = basis

        self.cart = False

    @classmethod
    def from_mol(cls, mol: Mol, xtb_version: str = "gfn1", **kwargs) -> PyscfMol:
        return cls(mol.numbers, mol.positions, xtb_version, **kwargs)

    def undo_norm_prim(self) -> None:
        """
        Circumvent normalization of primitive Gaussians, which is automatically
        done in the build step of the PySCF molecule.

        See: https://github.com/pyscf/pyscf/issues/1800.
        """
        bas_id = 0
        for at in self.atom:
            sym = at[0]
            assert isinstance(sym, str)
            basis_add = self.basis[sym]

            for b in basis_add:
                b_coeff = np.array(sorted(list(b[1:]), reverse=True))
                expnts = b_coeff[:, 0]
                coeffs = b_coeff[:, 1]

                e_adr = self._bas[bas_id, gto.PTR_EXP]
                c_adr = self._bas[bas_id, gto.PTR_COEFF]
                nprim = self.bas_nprim(bas_id)

                self._env[e_adr : e_adr + nprim] = expnts
                self._env[c_adr : c_adr + nprim] = coeffs

                bas_id += 1
