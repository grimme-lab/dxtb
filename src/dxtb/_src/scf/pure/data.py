"""
Data
====

This module defines the `_Data` class which is used for storing data for
single point calculations. This data includes properties such as atomic
numbers, core Hamiltonian, overlap matrix, occupation numbers, index mapping
for the basis set, restart data for interactions, electronic energy,
self-consistent Hamiltonian, density matrix, orbital energies (eigenvalues of
Fock matrix), LCAO coefficients (eigenvectors of Fock matrix), and the
iteration number.
"""

from __future__ import annotations

import torch

from dxtb import IndexHelper
from dxtb._src.components.interactions import InteractionListCache
from dxtb._src.components.interactions.container import ContainerData
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.typing import Slicers, Tensor

__all__ = ["_Data"]


class _Data:
    """
    Class for storing restart data for single point calculations.

    Attributes
    ----------
    numbers : Tensor
        Atomic numbers.
    occupation : Tensor
        Occupation numbers (shape: [..., 2, orbs]).
    nel : Tensor
        Number of alpha and beta electrons (shape: [..., 2]). Fixed during
        the SCF.
    n0 : Tensor
        Reference occupation for each orbital (shape: [..., orbs]).
    ihelp : IndexHelper
        Index mapping for the basis set.
    cache : InteractionListCache
        Restart data for the interactions.
    energy : Tensor
        Electronic energy (shape: [..., orbs]).
    hamiltonian : Tensor
        Self-consistent Hamiltonian (shape: [..., orbs, orbs]).
    density : Tensor
        Density matrix.
    evals : Tensor
        Orbital energies, i.e., eigenvalues of Fock matrix (shape: [..., orbs]).
    evecs : Tensor
        LCAO coefficients, i.e., eigenvectors of Fock matrix
        (shape: [..., orbs, orbs]).
    iter : int
        Number of iterations.
    """

    def __init__(
        self,
        occupation: Tensor,
        n0: Tensor,
        numbers: Tensor,
        ihelp: IndexHelper,
        cache: InteractionListCache,
        integrals: IntegralMatrices,
        nel: Tensor | None = None,
    ) -> None:
        """
        Initialize the _Data object.

        Parameters
        ----------
        hcore : Tensor
            Core Hamiltonian.
        overlap : Tensor
            Overlap matrix.
        occupation : Tensor
            Occupation numbers.
        nel : Tensor | None
            Number of alpha and beta electrons. Defaults to the sum of the
            occupation numbers.
        n0 : Tensor
            Reference occupation for each orbital.
        numbers : Tensor
            Atomic numbers.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : InteractionListCache
            Restart data for the interactions.
        """
        if integrals.hcore is None:
            raise ValueError("No core Hamiltonian provided.")
        if integrals.overlap is None:
            raise ValueError("No Overlap provided.")

        self.ints = integrals
        self.occupation = occupation
        self.nel = occupation.sum(-1) if nel is None else nel
        self.n0 = n0
        self.numbers = numbers
        self.ihelp = ihelp
        self.cache = cache
        self.init_zeros()

        self.potential: ContainerData = {
            "mono": None,
            "dipole": None,
            "quad": None,
            "label": None,
        }
        self.charges: ContainerData = {
            "mono": None,
            "dipole": None,
            "quad": None,
            "label": None,
        }

        self.iter = -1  # bumped before printing, guess energy also printed

    def init_zeros(self) -> None:
        """
        Initialize the energy, Hamiltonian, density, evals, evecs,
        old_charges, old_energy, and old_density attributes with zeros.
        """
        self.energy = torch.zeros_like(self.n0)
        self.hamiltonian = torch.zeros_like(self.ints.hcore)
        self.density = torch.zeros_like(self.ints.hcore)
        self.evals = torch.zeros_like(self.n0)
        self.evecs = torch.zeros_like(self.ints.hcore)

        self.old_charges = torch.zeros_like(self.energy)
        self.old_energy = torch.zeros_like(self.numbers)
        self.old_density = torch.zeros_like(self.density)

    def reset(self) -> None:
        """
        Reset the iteration count and reinitialize the attributes with zeros.
        """
        self.iter = 0
        self.init_zeros()

    def cull(self, conv: Tensor, slicers: Slicers) -> None:
        """
        Update the tensor attributes based on the given conditions.

        Parameters
        ----------
        conv : Tensor
            Condition tensor.
        slicers : Slicers
            Slicer objects for selecting data from tensors.
        """
        onedim = tuple([~conv, *slicers["orbital"]])
        onedim_atom = tuple([~conv, *slicers["atom"]])
        twodim = tuple([~conv, *slicers["orbital"], *slicers["orbital"]])
        threedim = tuple(
            [~conv, (...), *slicers["orbital"], *slicers["orbital"]]
        )

        # disable shape check temporarily for writing culled versions back
        self.ints.run_checks = False
        self.ints.overlap = self.ints.overlap[twodim]
        self.ints.hcore = self.ints.hcore[twodim]
        if self.ints.dipole is not None:
            self.ints.dipole = self.ints.dipole[threedim]
        if self.ints.quadrupole is not None:
            self.ints.quadrupole = self.ints.quadrupole[threedim]
        self.ints.run_checks = True

        self.numbers = self.numbers[onedim_atom]
        self.hamiltonian = self.hamiltonian[twodim]
        self.density = self.density[twodim]
        self.occupation = self.occupation[twodim]
        self.nel = self.nel[~conv]
        self.evecs = self.evecs[twodim]
        self.evals = self.evals[onedim]
        self.energy = self.energy[onedim]
        self.n0 = self.n0[onedim]
        self.ihelp.cull(conv, slicers=slicers)
        self.cache.cull(conv, slicers=slicers)

        self.old_charges = self.old_charges[onedim]
        self.old_energy = self.old_energy[onedim_atom]
        self.old_density = self.old_density[twodim]
