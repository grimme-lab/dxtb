"""
Self-consistent field iteration
===============================

Provides implementation of self consistent field iterations for the xTB Hamiltonian.
The iterations are not like in ab initio calculations expressed in the density matrix
and the derivative of the energy w.r.t. to the density matrix, i.e. the Hamiltonian,
but the Mulliken populations (or partial charges) of the respective orbitals as well
as the derivative of the energy w.r.t. to those populations, i.e. the potential vector.

The implementation is based on the `xitorch <https://xitorch.readthedocs.io>`__ library,
which appears to be abandoned and unmaintained at the time of writing, but still provides
a reasonably good implementation of the iterative solver required for the self-consistent
field iterations.
"""

from __future__ import annotations
import math
import torch
import xitorch as xt
import xitorch.linalg as xtl
import xitorch.optimize as xto

from ..basis import IndexHelper
from ..interaction import Interaction
from ..typing import Any, Tensor
from ..utils import t2int


class SelfConsistentField(xt.EditableModule):
    """
    Self-consistent field iterator, which can be used to obtain a self-consistent
    solution for a given Hamiltonian.
    """

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

        cache: Interaction.Cache
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
            cache: Interaction.Cache,
        ):

            self.hcore = hcore
            self.overlap = overlap
            self.occupation = occupation
            self.n0 = n0
            self.ihelp = ihelp
            self.cache = cache
            self.energy = hcore.new_tensor(0.0)

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

    use_potential: bool
    """Whether to use the potential or the charges"""

    def __init__(
        self,
        interaction: Interaction,
        *args,
        **kwargs,
    ):
        self.use_potential = kwargs.pop("use_potential", False)
        self.bck_options = {"posdef": True, **kwargs.pop("bck_options", {})}

        self.fwd_options = {
            "method": "broyden1",
            "alpha": -0.5,
            "f_tol": 1.0e-5,
            "x_tol": 1.0e-5,
            "f_rtol": float("inf"),
            "x_rtol": float("inf"),
            "maxiter": 50,
            "verbose": False,
            "line_search": False,
            **kwargs.pop("fwd_options", {}),
        }

        self.eigen_options = {"method": "exacteig", **kwargs.pop("eigen_options", {})}

        self._data = self._Data(*args, **kwargs)
        self.interaction = interaction

    def __call__(self, charges: Tensor | None = None) -> dict[str, Tensor]:
        """
        Run the self-consistent iterations until a stationary solution is reached

        Parameters
        ----------
        charges : Tensor, optional
            Initial orbital charges vector.

        Returns
        -------
        Tensor
            Converged orbital charges vector.
        """

        if charges is None:
            charges = torch.zeros_like(self._data.occupation)

        print("guess", self.charges_to_potential(charges))
        output = xto.equilibrium(
            fcn=self.iterate_potential if self.use_potential else self.iterate_charges,
            y0=self.charges_to_potential(charges) if self.use_potential else charges,
            bck_options={**self.bck_options},
            **self.fwd_options,
        )

        charges = self.potential_to_charges(output) if self.use_potential else output
        energy = self.get_energy(charges)
        return {
            "charges": charges,
            "energy": energy,
            "density": self._data.density,
            "hamiltonian": self._data.hamiltonian,
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

        if self.fwd_options["verbose"] > 1:
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
        print("\niterate pot: charges", charges)
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
        Obtain the density matrix from the potential.

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
        Compute the orbital charges from the density matrix.

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
            torch.diagonal(
                torch.einsum("...ik,...kj->...ij", density, self._data.hcore),
                dim1=-2,
                dim2=-1,
            )
        )

        populations = torch.diagonal(
            torch.einsum("...ik,...kj->...ij", density, self._data.overlap),
            dim1=-2,
            dim2=-1,
        )
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
        Compute the density matrix from the Hamiltonian.

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
        o_op = self.get_overlap()
        evals, evecs = self.diagonalize(h_op, o_op)

        nel = self._data.occupation.sum(-1)

        # electronic temp
        kt = 9.5004257356353500e-004

        maxiter = 200
        thr = math.sqrt(torch.finfo(self.dtype).eps)

        # indexing starts with zero -> -1
        homo = t2int(torch.floor(nel / 2))
        homo = homo + 1 if nel / 2 % 1 > 0.5 else homo
        occt = homo

        e_fermi = 0.5 * (evals[max(homo - 1, 0)] + evals[min(homo, evals.size(0) - 1)])

        # print("nel", nel, "homo", homo)
        print("evals", evals)
        # print("efermi", e_fermi)

        occ = torch.zeros_like(self._data.occupation)
        eps = torch.tensor(torch.finfo(self.dtype).eps, device=self.device)

        for iter in range(maxiter):
            total_number = 0.0
            total_dfermi = 0.0

            for iao in range(evals.size(0)):
                fermifunc = 0.0
                if (evals[iao] - e_fermi) / kt < 50:
                    fermifunc = 1.0 / (torch.exp((evals[iao] - e_fermi) / kt) + 1.0)
                    dfermifunc = torch.exp((evals[iao] - e_fermi) / kt) / (
                        kt * (torch.exp((evals[iao] - e_fermi) / kt) + 1.0) ** 2
                    )
                    fermifunc = float(fermifunc)
                    dfermifunc = float(dfermifunc)
                else:
                    dfermifunc = 0.0

                occ[iao] = fermifunc
                total_number += fermifunc
                total_dfermi += dfermifunc

                # print(iao, occ[iao], evals[iao], e_fermi)

            change_fermi = (occt - total_number) / (total_dfermi + eps)
            # print("total_number", total_number, occt, total_dfermi, change_fermi)
            e_fermi = e_fermi + change_fermi

            if abs(occt - total_number) <= thr:
                break

        # print("occ", occ)
        # print("evals", evals)
        # print("self._data.occupation\n", self._data.occupation)
        # self._data.occupation = occ * 2
        ret = torch.einsum(
            "...ik,...k,...kj->...ij",
            evecs,
            self._data.occupation,
            evecs.mT,
        )
        print("denisty", ret)
        return ret

    def get_overlap(self) -> xt.LinearOperator:
        """
        Get the overlap matrix.

        Returns
        -------
        xt.LinearOperator
            Overlap matrix.
        """

        smat = self._data.overlap

        zeros = torch.eq(smat, 0)
        mask = torch.all(zeros, dim=-1) & torch.all(zeros, dim=-2)

        return xt.LinearOperator.m(
            smat + torch.diag_embed(smat.new_ones(*smat.shape[:-2], 1) * mask)
        )

    def diagonalize(
        self, hamiltonian: xt.LinearOperator, overlap: xt.LinearOperator
    ) -> tuple[Tensor, Tensor]:
        """
        Diagonalize the Hamiltonian.

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
            return [prefix + "_data.hcore"] + self.getparamnames(
                "diagonalize", prefix=prefix
            )
        if methodname == "potential_to_hamiltonian":
            return [prefix + "_data.hcore", prefix + "_data.overlap"]
        if methodname == "diagonalize":
            return [prefix + "_data.overlap"]
        raise KeyError(f"Method '{methodname}' has no paramnames set")


def solve(
    numbers: Tensor,
    positions: Tensor,
    interactions: Interaction,
    ihelp: IndexHelper,
    *args,
    **kwargs,
) -> dict[str, Tensor]:
    """
    Obtain self-consistent solution for a given Hamiltonian.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the system.
    positions : Tensor
        Positions of the system.
    interactions : Interaction
        Interaction object.
    ihelp : IndexHelper
        Index helper object.
    args : Tuple
        Positional arguments to pass to the engine.
    kwargs : Dict
        Keyword arguments to pass to the engine.

    Returns
    -------
    Tensor
        Orbital-resolved partial charges vector.
    """

    cache = interactions.get_cache(numbers, positions, ihelp)
    return SelfConsistentField(
        interactions, *args, ihelp=ihelp, cache=cache, **kwargs
    )()
