# This file is part of xtbml.

"""
Base calculator for the extended tight-binding model.
"""

from __future__ import annotations
import torch

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
from ..wavefunction import filling
from ..typing import Tensor
from ..coulomb import secondorder, thirdorder, averaging_function
from ..param import get_element_param, get_elem_param_dict


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

    interaction: Interaction
    """Interactions to minimize in self-consistent iterations."""

    def __init__(self, mol: Geometry, par: Param, acc: float = 1.0) -> None:

        self.basis = Basis(mol, par, acc)
        self.hamiltonian = Hamiltonian(mol, par)

        es2 = secondorder.ES2(
            hubbard=get_element_param(par.element, "gam"),
            lhubbard=get_elem_param_dict(par.element, "lgam"),
            average=averaging_function[par.charge.effective.average],
            gexp=torch.tensor(par.charge.effective.gexp),
        )
        es3 = thirdorder.ES3(
            hubbard_derivs=get_element_param(par.element, "gam3")[mol.atomic_numbers],
        )
        self.interaction = InteractionList([es2, es3])

    def singlepoint(
        self,
        mol: Geometry,
        ihelp: IndexHelper,
        verbosity: int = 1,
    ) -> dict[str, Tensor]:
        """
        Entry point for performing single point calculations.

        Parameters
        ----------
        mol : Geometry
            Molecular structure data.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Atom resolved energies.
        """

        rcov = covalent_rad_d3[mol.atomic_numbers]
        cn = ncoord.get_coordination_number(
            mol.atomic_numbers, mol.positions, ncoord.exp_count, rcov
        )

        cutoff = get_cutoff(self.basis)
        adjlist = AdjacencyList(mol, cutoff)

        hcore, overlap = self.hamiltonian.build(self.basis, adjlist, cn)

        # Obtain the reference occupations and total number of electrons
        n0 = self.hamiltonian.get_occupation(ihelp)
        nel = torch.sum(n0, -1) - torch.sum(mol.charges, -1)
        focc = 2 * filling.get_aufbau_occupation(
            hcore.new_tensor(hcore.shape[-1], dtype=torch.int64),
            nel / 2,
        )

        fwd_options = {
            "verbose": verbosity > 0,
        }
        results = scf.solve(
            mol.atomic_numbers,
            mol.positions,
            self.interaction,
            ihelp,
            hcore,
            overlap,
            focc,
            n0,
            fwd_options=fwd_options,
            use_potential=True,
        )

        return results
