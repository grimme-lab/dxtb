"""
"""

from __future__ import annotations
import torch
import xitorch as xt, xitorch.optimize as xto, xitorch.linalg as xtl

from ..interaction import Interaction
from ..basis import IndexHelper
from ..typing import Any, Tensor 


class SelfConsistentCharges(xt.EditableModule):
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

        energy: Tensor
        """Electronic energy"""

        hamiltonian: Tensor
        """Self-consistent Hamiltonian"""

        density: Tensor
        """Density matrix"""

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

    _data: "_Data"
    """Persistent data"""

    interaction: Interaction
    """Interactions to minimize in self-consistent iterations"""

    fwd_options: dict[str, Any]
    """Options for forwards pass"""

    bck_options: dict[str, Any]
    """Options for backwards pass"""

    eigen_options: dict[str, Any]
    """Options for eigensolver"""

    def __init__(
        self,
        interaction: Interaction,
        *args,
        **kwargs,
    ):
        self.bck_options = {"posdef": True, **kwargs.pop("bck_options", {})}

        self.fwd_options = {
            "method": "broyden1",
            "alpha": -0.5,
            "maxiter": 50,
            "verbose": False,
            **kwargs.pop("fwd_options", {}),
        }

        self.eigen_options = {"method": "exacteig", **kwargs.pop("eigen_options", {})}

        self._data = self._Data(*args, **kwargs)
        self.interaction = interaction

    def equilibrium(
        self, charges: Tensor | None = None, use_potential: bool = False
    ) -> dict[str, Tensor]:
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
            charges = torch.zeros_like(self._data.occupation)

        output = xto.equilibrium(
            fcn=self.iterate_potential if use_potential else self.iterate_charges,
            y0=self.charges_to_potential(charges) if use_potential else charges,
            bck_options={**self.bck_options},
            **self.fwd_options,
        )

        charges = self.potential_to_charges(output) if use_potential else output
        energy = self.get_energy(charges)
        return {
            "energy": energy,
            "charges": charges,
            "hamiltonian": self._data.hamiltonian,
            "density": self._data.density,
        }

    def get_energy(self, charges: Tensor) -> Tensor:
        """
        Get the energy of the system with the given charges.

        Parameters
        ----------
        charges : Tensor
            Orbital charges vector.

        Returns
        -------
        Tensor
            Energy of the system.
        """
        return self._data.energy + self.interaction.get_energy(
            charges, self._data.ihelp, self._data.cache
        )

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

        if self.fwd_options["verbose"]:
            print(self.get_energy(charges).sum(-1))
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
        if self.fwd_options["verbose"]:
            print(self.get_energy(charges).sum(-1))
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

        return self.interaction.get_potential(
            charges, self._data.ihelp, self._data.cache
        )

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

        self._data.density = self.potential_to_density(potential)
        return self.density_to_charges(self._data.density)

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

        self._data.hamiltonian = self.potential_to_hamiltonian(potential)
        return self.hamiltonian_to_density(self._data.hamiltonian)

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

        self._data.energy = self._data.ihelp.reduce_orbital_to_atom(
            torch.diagonal(density @ self._data.hcore)
        )

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
        evals, evecs = self.diagonalize(h_op, self._data.overlap)
        return (
            evecs @ torch.diag_embed(self._data.occupation, dim1=-2, dim2=-1) @ evecs.mT
        )

    def get_overlap(self) -> xt.LinearOperator:
        """
        Get the overlap matrix.

        Returns
        -------
        xt.LinearOperator
            Overlap matrix.
        """

        return xt.LinearOperator.m(self._data.overlap)

    def diagonalize(
        self, hamiltonian: xt.LinearOperator, overlap: xt.LinearOperator
    ) -> tuple[Tensor, Tensor]:
        """
        Diagnalize the Hamiltonian.

        Parameters
        ----------
        hamiltonian : xt.LinearOperator
            Current Hamiltonian matrix.
        overlap : xt.LinearOperator
            Overlap matrix.

        Returns
        -------
        evals : Tensor
            Eigenvalues of the Hamiltonian.
        evecs : Tensor
            Eigenvectors of the Hamiltonian.
        """

        overlap = self.get_overlap()
        return xtl.lsymeig(A=hamiltonian, M=overlap, **self.eigen_options)

    @property
    def shape(self):
        """
        Returns the shape of the density matrix in this engine.
        """
        return self._data.hcore.shape

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the tensors in this engine.
        """
        return self._data.hcore.dtype

    @property
    def device(self) -> torch.device:
        """
        Returns the device of the tensors in this engine.
        """
        return self._data.hcore.device

    def getparamnames(self, methodname: str, prefix: str = "") -> list[str]:
        if methodname == "iterate_charges":
            return self.getparamnames(
                "charges_to_potential", prefix=prefix
            ) + self.getparamnames("potential_to_charges", prefix=prefix)
        if methodname == "iterate_potential":
            return self.getparamnames(
                "potential_to_charges", prefix=prefix
            ) + self.getparamnames("charges_to_potential", prefix=prefix)
        if methodname == "charges_to_potential":
            return []
        if methodname == "potential_to_charges":
            return self.getparamnames(
                "potential_to_density", prefix=prefix
            ) + self.getparamnames("density_to_charges", prefix=prefix)
        if methodname == "potential_to_density":
            return self.getparamnames(
                "potential_to_hamiltonian", prefix=prefix
            ) + self.getparamnames("hamiltonian_to_density", prefix=prefix)
        if methodname == "density_to_charges":
            return []
        if methodname == "hamiltonian_to_density":
            return [prefix + "_data.hcore"]
        if methodname == "potential_to_hamiltonian":
            return [prefix + "_data.hcore", prefix + "_data.overlap"]
        if methodname == "diagonalize":
            return [prefix + "_data.overlap"]
        raise KeyError("Method %s has no paramnames set" % methodname)
