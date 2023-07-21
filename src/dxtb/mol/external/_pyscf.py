"""
Representation of Molecule in PySCF
===================================

This module contains a class for a molecular representation in PySCF's format.
It also contains the short-cut version as in PySCF (``M``).

Example
-------
>>> import torch
>>> from dxtb.mol.external._pyscf import M
>>>
>>> numbers = torch.tensor([14, 1, 1, 1, 1])
>>> positions = torch.tensor([
...     [+0.00000000000000, +0.00000000000000, +0.00000000000000],
...     [+1.61768389755830, +1.61768389755830, -1.61768389755830],
...     [-1.61768389755830, -1.61768389755830, -1.61768389755830],
...     [+1.61768389755830, -1.61768389755830, +1.61768389755830],
...     [-1.61768389755830, +1.61768389755830, +1.61768389755830],
... ])
>>> mol = M(numbers, positions)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # pragma: no cover
    from pyscf import gto  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("PySCF is not installed") from e

from ..._types import Tensor
from ...constants import PSE
from ...utils import tensor_to_numpy
from ..molecule import Mol

# Turn off PySCF's normalization since dxtb's normalization is different,
# requiring a separate normalization anyway. But this way, the results are
# equal to dxtb's libcint wrapper and we can immediately compare integrals.
gto.mole.NORMALIZE_GTO = False

__all__ = ["PyscfMol", "M"]


def M(
    numbers: Tensor, positions: Tensor, xtb_version: str = "gfn1", **kwargs
) -> PyscfMol:
    """
    Shortcut for building up the `PyscfMol`.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms in the system.
    positions : Tensor
        Cartesian coordinates of all atoms in the system.
    xtb_version : str, optional
        Version of xtb to use for basis set. Defaults to "gfn1".

    Returns
    -------
    PyscfMol
        Built-up PySCF molecule.
    """
    mol = PyscfMol(numbers, positions, xtb_version=xtb_version)
    mol.build(**kwargs)
    return mol


class PyscfMol(gto.Mole):
    """
    Pyscf's molecule representation that can be created upon passing only
    `numbers` and `positions`. Note that the basis set is created from a
    database and only the xtb version must be provided during instantiation.

    .. warning::

        You still have to initialize the molecule via `mol.build()`.
    """

    def __init__(
        self, numbers: Tensor, positions: Tensor, xtb_version: str = "gfn1", **kwargs
    ):
        # init pyscf's molecule type
        super().__init__(**kwargs)

        # TODO: Check if file exists ()
        # path to basis set storage
        path = Path(Path(__file__).resolve().parent, "basis", xtb_version)

        # internal format of `pyscf.gto.Mole.atom`
        # atom = [[atom1, (x, y, z)],
        #         [atom2, (x, y, z)],
        #         ...
        #         [atomN, (x, y, z)]]
        atom: list[list[str | np.ndarray]] = []

        basis: dict[str, list] = {}
        for i, number in enumerate(numbers.tolist()):
            if number not in PSE:
                raise ValueError(f"Atom '{number}' not found.")

            symbol = PSE[number]
            pos = tensor_to_numpy(positions[i, :])
            atom.append([symbol, pos])

            # pyscf expects nwchem format of basis set
            p = Path(path, f"{number:02d}.nwchem")
            with open(p, encoding="utf8") as f:
                content = f.read()
                basis[f"{PSE[number]}"] = gto.parse(content)

        self.atom = atom
        self.basis = basis
        self.unit = "B"  # unit: Bohr (a.u.)
        self.cart = False

    @classmethod
    def from_mol(cls, mol: Mol, xtb_version: str = "gfn1", **kwargs) -> PyscfMol:
        return cls(mol.numbers, mol.positions, xtb_version, **kwargs)
