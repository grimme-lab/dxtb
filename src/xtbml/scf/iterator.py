"""
"""

import torch
import xitorch as xt

from .interaction import Interaction
from ..basis import IndexHelper
from ..typing import Tensor, Dict


class SelfConsistentCharges:
    """
    """

    class Cache:
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

        def __init__(
            self,
            interaction: Interaction,
            hcore: Tensor,
            overlap: Tensor,
            occupation: Tensor,
            n0: Tensor,
            ihelp: IndexHelper,
        ):

            self.hcore = hcore
            self.overlap = overlap
            self.occupation = occupation
            self.n0 = n0
            self.ihelp = ihelp

    cache: self.Cache
    """"""

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

        self.= self.Cache(*args, **kwargs)
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

    def equilibrium(self, density: Optional[Tensor] = None) -> Tensor:
        """
        Run the self-consisten iterations until a stationary solution is reached

        Parameters
        ----------
        density : Tensor, optional
            Initial density matrix.

        Returns
        -------
        Tensor
            Converged density matrix.
        """

        hamiltonian0 = (
            self.cache.hcore
            if density is None
            else self.density_to_hamiltonian(density)
        )

        hamiltonian = xt.optimize.equilibrium(
            fcn=self.iteration,
            y0=hamiltonian0,
            bck_options={**self.bck_options},
            **fwd_options,
        )

        return self.hamiltonian_to_density(hamiltonian)

    def iteration(self, potential: Tensor):
        """
        Perform single self-consistent iteration.

        Parameters
        ----------
        potential : Tensor
            Potential vector for each orbital partial charge.

        Returns
        -------
        Tensor
            New potential vector.
        """

        density = self.potential_to_density(potential)
        return self.density_to_potential(density)

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
        density = self.hamiltonian_to_density(hamiltonian)

    def density_to_potential(self, density: Tensor):
        """
        Obtain the Hamiltonian from the density matrix.

        Parameters
        ----------
        density : Tensor
            Density matrix.

        Returns
        -------
        Tensor
            Potential vector for each orbital partial charge.
        """

        populations = torch.diagonal(density @ self.cache.overlap, dim1=-2, dim2=-1)
        charges = self.cache.n0 - populations

        return self.interaction.get_potential(charges, self.cache.ihelp)


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

        return (
            self.cache.hcore
            - 0.5 * self.cache.overlap * (potential.unsqueeze(-1) + potential.unsqueeze(-2))
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
        evals, evecs = self.diagnalize(h_op, self.cache.overlap)
        return evecs @ self.cache.occupation @ evecs.mT

    def get_overlap(self) -> xt.LinearOperator:
        """
        Get the overlap matrix.

        Returns
        -------
        xt.LinearOperator
            Overlap matrix.
        """

        return xt.LinearOperator.m(self.cache.overlap)

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
