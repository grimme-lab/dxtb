# This file is part of xtbml.

"""
Base calculator for the extended tight-binding model.
"""

from __future__ import annotations
import torch

from .. import scf
from ..basis.type import Basis
from ..basis import IndexHelper
from ..coulomb import secondorder, thirdorder, averaging_function
from ..data.covrad import covalent_rad_d3
from ..interaction import Interaction, InteractionList
from ..ncoord import ncoord
from ..param import Param, get_element_param, get_elem_param_dict
from ..typing import Tensor
from ..wavefunction import filling
from ..xtb.h0 import Hamiltonian


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

    def __init__(
        self, numbers: Tensor, positions: Tensor, par: Param, acc: float = 1.0
    ) -> None:

        self.basis = Basis(numbers, par, acc)
        self.hamiltonian = Hamiltonian(numbers, positions, par)

        if par.charge is None:
            raise ValueError("No charge parameters provided.")

        es2 = secondorder.ES2(
            hubbard=get_element_param(par.element, "gam"),
            lhubbard=get_elem_param_dict(par.element, "lgam"),
            average=averaging_function[par.charge.effective.average],
            gexp=torch.tensor(par.charge.effective.gexp),
        )
        es3 = thirdorder.ES3(
            hubbard_derivs=get_element_param(par.element, "gam3")[numbers],
        )
        self.interaction = InteractionList([es2, es3])

    def singlepoint(
        self,
        numbers: Tensor,
        positions: Tensor,
        charges: Tensor,
        ihelp: IndexHelper,
        verbosity: int = 1,
    ) -> dict[str, Tensor]:
        """
        Entry point for performing single point calculations.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers.
        positions : Tensor
            Atomic positions.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Atom resolved energies.
        """

        rcov = covalent_rad_d3[numbers]
        cn = ncoord.get_coordination_number(numbers, positions, ncoord.exp_count, rcov)

        overlap = self.hamiltonian.overlap()
        hcore = self.hamiltonian.build(overlap, cn)

        # Obtain the reference occupations and total number of electrons
        n0 = self.hamiltonian.get_occupation(ihelp)
        nel = torch.sum(n0, -1) - torch.sum(charges, -1)
        occupation = 2 * filling.get_aufbau_occupation(
            hcore.new_tensor(hcore.shape[-1], dtype=torch.int64),
            nel / 2,
        )

        fwd_options = {
            "verbose": verbosity > 1,
        }
        results = scf.solve(
            numbers,
            positions,
            self.interaction,
            ihelp,
            hcore,
            overlap,
            occupation,
            n0,
            fwd_options=fwd_options,
            use_potential=True,
        )

        return results
