"""
"""

import torch
import xitorch as xt

from .interaction import Interaction
from ..basis import IndexHelper
from ..typing import Tensor, Dict


class SelfConsistentCharges:
    """ """

    class _Data:
        """
        Restart data for the singlepoint calculation.
        """

        hcore: Tensor
        """Core Hamiltonian"""

        overlap: Tensor
        """Overlap matrix"""

        occupation: Tensor
        """Occupation numbers"""

        n0: Tensor
        """Reference occupations for each orbital"""

        ihelp: IndexHelper
        """Index mapping for the basis set"""

        cache: "Cache"
        """Restart data for the interaction"""

        def __init__(
            self,
            hcore: Tensor,
            overlap: Tensor,
            occupation: Tensor,
            n0: Tensor,
            ihelp: IndexHelper,
            cache: "Cache",
        ):

            self.hcore = hcore
            self.overlap = overlap
            self.occupation = occupation
            self.n0 = n0
            self.ihelp = ihelp
            self.cache = cache

    _data: self._Data
    """Persistent data"""

    interaction: Interaction
    """Interactions to minimize in self-consistent iterations"""

    fwd_options: Dict[str, ...]
    """Options for forwards pass"""

    bck_options: Dict[str, ...]
    """Options for backwards pass"""

    eigen_options: Dict[str, ...]
    """Options for eigensolver"""

    def __init__(
        self,
        interaction: Interaction,
        *args,
        **kwargs,
    ):

        self._date = self._Data(*args, **kwargs)
        self.interaction = interaction

        self.bck_options = {
            "posdef": True,
        }

        self.fwd_options = {
            "method": "broyden1",
            "alpha": -0.5,
            "maxiter": 50,
            "verbose": False,
        }

        self.eigen_options = {
            "method": "exacteig",
        }

    def equilibrium(
        self, charges: Optional[Tensor] = None, use_potential: bool = False
    ) -> Tensor:
        """
        Run the self-consisten iterations until a stationary solution is reached

        Parameters
        ----------
        charges : Tensor, optional
            Initial orbital charges vector.
        use_potential : bool, optional
            Iterate using the potential instead of the charges.

        Returns
        -------
        Tensor
            Converged orbital charges vector.
        """

        if charges is None:
            torch.zeros_like(self._data.occupation)

        output = xt.optimize.equilibrium(
            fcn=self.iterate_potential if use_potential else self.iterate_charges,
            y0=self.charges_to_potential(charges) if use_potential else charges,
            bck_options={**self.bck_options},
            **fwd_options,
        )

        return self.potential_to_charges(output) if use_potential else output

    def iterate_charges(self, charges: Tensor):
        """
        Perform single self-consistent iteration.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges vector.

        Returns
        -------
        Tensor
            New orbital-resolved partial charges vector.
        """

        potential = self.charges_to_potential(charges)
        return self.potential_to_charges(potential)

    def iterate_potential(self, potential: Tensor):
        """
        Perform single self-consistent iteration.

        Parameters
        ----------
        potential: Tensor
            Potential vector for each orbital partial charge.

        Returns
        -------
        Tensor
            New potential vector for each orbital partial charge.
        """

        charges = self.potential_to_charges(potential)
        return self.charges_to_potential(charges)

    def charges_to_potential(self, charges: Tensor) -> Tensor:
        """
        Compute the potential from the orbital charges.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges vector.

        Returns
        -------
        Tensor
            Potential vector for each orbital partial charge.
        """

        return self.interaction.get_potential(charges, self._data.ihelp, self._data.cache)

    def potential_to_charges(self, potential: Tensor) -> Tensor:
        """
        Compute the orbital charges from the potential.

        Parameters
        ----------
        potential : Tensor
            Potential vector for each orbital partial charge.

        Returns
        -------
        Tensor
            Orbital-resolved partial charges vector.
        """

        density = self.potential_to_density(potential)
        return self.density_to_charges(density)

    def potential_to_density(self, potential: Tensor):
        """
        Obtain the density matrix from the Hamiltonian.

        Parameters
        ----------
        potential : Tensor
            Potential vector for each orbital partial charge.

        Returns
        -------
        Tensor
            Density matrix.
        """

        hamiltonian = self.potential_to_hamiltonian(potential)
        return self.hamiltonian_to_density(hamiltonian)

    def density_to_charges(self, density: Tensor):
        """
        Obtain the Hamiltonian from the density matrix.

        Parameters
        ----------
        density : Tensor
            Density matrix.

        Returns
        -------
        Tensor
            Orbital-resolved partial charges vector.
        """

        populations = torch.diagonal(density @ self._data.overlap, dim1=-2, dim2=-1)
        return self._data.n0 - populations

    def potential_to_hamiltonian(self, potential: Tensor) -> Tensor:
        """
        Compute the Hamiltonian from the potential.

        Parameters
        ----------
        potential : Tensor
            Potential vector for each orbital partial charge.

        Returns
        -------
        Tensor
            Hamiltonian matrix.
        """

        return self._data.hcore - 0.5 * self._data.overlap * (
            potential.unsqueeze(-1) + potential.unsqueeze(-2)
        )

    def hamiltonian_to_density(self, hamiltonian: Tensor):
        """
        Compute density matrix from the Hamiltonian.

        Parameters
        ----------
        hamiltonian : Tensor
            Hamiltonian matrix.

        Returns
        -------
        Tensor
            Density matrix.
        """

        h_op = xt.LinearOperator.m(hamiltonian)
        evals, evecs = self.diagnalize(h_op, self._data.overlap)
        return evecs @ self._data.occupation @ evecs.mT

    def get_overlap(self) -> xt.LinearOperator:
        """
        Get the overlap matrix.

        Returns
        -------
        xt.LinearOperator
            Overlap matrix.
        """

        return xt.LinearOperator.m(self._data.overlap)

    def diagnalize(self, hamiltonian: xt.LinearOperator):
        """
        Diagnalize the Hamiltonian.

        Parameters
        ----------
        hamiltonian : Tensor
            Current Hamiltonian matrix.

        Returns
        -------
        evals : Tensor
            Eigenvalues of the Hamiltonian.
        evecs : Tensor
            Eigenvectors of the Hamiltonian.
        """

        overlap = self.get_overlap()
        return xt.linalg.lsymeig(A=hamiltonian, M=overlap, **self.eigen_options)
