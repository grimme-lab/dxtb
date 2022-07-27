# This file is part of xtbml.

"""
Base calculator for the extended tight-binding model.
"""

from __future__ import annotations
import torch

from xtbml.param.util import get_elem_param, get_element_angular

from .. import scf
from ..basis import Basis, IndexHelper
from ..coulomb import secondorder, thirdorder, averaging_function
from ..data import cov_rad_d3
from ..interaction import Interaction, InteractionList
from ..ncoord import ncoord
from ..param import Param
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

    ihelp: IndexHelper
    """Helper class for indexing."""

    def __init__(
        self, numbers: Tensor, positions: Tensor, par: Param, acc: float = 1.0
    ) -> None:
        self.ihelp = IndexHelper.from_numbers(numbers, get_element_angular(par.element))
        self.basis = Basis(numbers, par, acc)
        self.hamiltonian = Hamiltonian(numbers, positions, par, self.ihelp)

        if par.charge is None:
            raise ValueError("No charge parameters provided.")

        es2 = secondorder.ES2(
            hubbard=get_elem_param(
                torch.unique(numbers),
                par.element,
                "gam",
                device=positions.device,
                dtype=positions.dtype,
            ),
            lhubbard=get_elem_param(
                torch.unique(numbers),
                par.element,
                "lgam",
                device=positions.device,
                dtype=positions.dtype,
            ),
            average=averaging_function[par.charge.effective.average],
            gexp=torch.tensor(par.charge.effective.gexp),
        )
        es3 = thirdorder.ES3(
            hubbard_derivs=get_elem_param(
                torch.unique(numbers),
                par.element,
                "gam3",
                device=positions.device,
                dtype=positions.dtype,
            ),
        )

        self.interaction = InteractionList([es2, es3])

    def singlepoint(
        self,
        numbers: Tensor,
        positions: Tensor,
        charges: Tensor,
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

        rcov = cov_rad_d3[numbers]
        cn = ncoord.get_coordination_number(numbers, positions, ncoord.exp_count, rcov)

        overlap = self.hamiltonian.overlap()
        overlap = self.hamiltonian.overlap_new()
        hcore = self.hamiltonian.build(overlap, cn)

        # Obtain the reference occupations and total number of electrons
        n0 = self.hamiltonian.get_occupation()
        nel = torch.sum(n0, -1) - torch.sum(charges, -1)
        occupation = 2 * filling.get_aufbau_occupation(
            hcore.new_tensor(hcore.shape[-1], dtype=torch.int64),
            nel / 2,
        )

        fwd_options = {
            "verbose": verbosity > 0,
        }
        results = scf.solve(
            numbers,
            positions,
            self.interaction,
            self.ihelp,
            hcore,
            overlap,
            occupation,
            n0,
            fwd_options=fwd_options,
            use_potential=True,
        )

        return results
