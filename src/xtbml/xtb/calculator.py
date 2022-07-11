# This file is part of xtbml.

"""
Base calculator for the extended tight-binding model.
"""

import torch
import xitorch as xt

from ..basis.type import Basis, get_cutoff
from ..basis import IndexHelper
from ..exlibs.tbmalt import Geometry
from ..param import Param
from ..xtb.h0 import Hamiltonian
from .. import scf
from ..interaction import Interaction, InteractionList
from ..adjlist import AdjacencyList
from ..ncoord import ncoord
from ..data.covrad import covalent_rad_d3
from ..wavefunction import mulliken


class Calculator:
    """
    Parametrized calculator defining the extended tight-binding model.

    The calculator holds the atomic orbital basis set for defining the Hamiltonian
    and the overlap matrix.
    """

    basis: Basis
    """Atomic orbital basis set definition."""

    hamiltonian: Hamiltonian
    """Core Hamiltonian definition."""

    cache = None
    """Restart info"""

    interaction: Interaction
    """Interactions to minimize in self-consistent iterations."""

    def __init__(self, mol: Geometry, par: Param, acc: float = 1.0) -> None:

        self.basis = Basis(mol, par, acc)
        self.hamiltonian = Hamiltonian(mol, par)
        self.interaction = InteractionList([])

    class Cache:
        """
        Restart data for the extended tight-binding model.
        """

        hcore: torch.Tensor
        """Core Hamiltonian."""

        overlap: torch.Tensor
        """Overlap matrix."""

    def get_cache(self) -> self.Cache:
        """
        Create restart data for the singlepoint calculation

        Returns
        -------
        Cache
            Restart data for the singlepoint calculation.
        """

        return self.Cache()

    def singlepoint(
        self,
        mol: Geometry,
        ihelp: IndexHelper,
    ) -> Tensor:
        """
        Entry point for performing single point calculations.

        Parameters
        ----------
        mol : Geometry
            Molecular structure data.
        cache : Cache
            Restart data for the singlepoint calculation.

        Returns
        -------
        Tensor
            Atom resolved energies.
        """

        solver = scf.solver.eigh
        # pot = scf.Potential()

        rcov = covalent_rad_d3[mol.numbers]
        cn = ncoord.get_coordination_number(
            mol.numbers, mol.positions, ncoord.exp_count, rcov
        )

        cutoff = get_cutoff(self.basis)
        adjlist = AdjacencyList(mol, cutoff)

        hcore, overlap = self.hamiltonian.build(self.basis, adjlist, cn)

        # Obtain the reference occupations and total number of electrons
        n0 = self.hamiltonian.get_occupation(self.basis)
        nel = torch.sum(n0, -1) - mol.charges
        focc = 2 * wavefunction.filling.get_aufbau_occupation(
            hcore.new_tensor(hcore.shape[-1], dtype=torch.int64),
            nel / 2,
        )

        scc = scf.SelfConsistentCharges(
            self.interaction, hcore, overlap, focc, n0, ihelp
        )
        density = scc.equilibrium()

        charges = n0 - torch.diagonal(density @ overlap, dim1=-2, dim2=-1)
        return self.interaction.get_energy(charges, ihelp)
