"""
Self-consistent field
=====================
"""
from __future__ import annotations

from abc import abstractmethod

import torch

from .._types import Any, Slicers, Tensor
from ..basis import IndexHelper
from ..constants import K2AU, defaults
from ..exlibs.xitorch import EditableModule, LinearOperator
from ..exlibs.xitorch import linalg as xtl
from ..interaction import InteractionList
from ..utils import real_atoms
from ..wavefunction import filling, mulliken


class BaseSelfConsistentField(EditableModule):
    """
    Self-consistent field iterator, which can be used to obtain a
    self-consistent solution for a given Hamiltonian.

    This base class only lacks the `scf` method, which implements mixing and
    convergence.
    """

    class _Data:
        """
        Restart data for the singlepoint calculation.
        """

        numbers: Tensor
        """Atomic numbers"""

        hcore: Tensor
        """Core Hamiltonian"""

        overlap: Tensor
        """Overlap matrix"""

        occupation: Tensor
        """Occupation numbers (shape: [..., 2, orbs])"""

        n0: Tensor
        """Reference occupations for each orbital (shape: [..., orbs])"""

        ihelp: IndexHelper
        """Index mapping for the basis set"""

        cache: InteractionList.Cache
        """Restart data for the interactions"""

        energy: Tensor
        """Electronic energy (shape: [..., orbs])"""

        hamiltonian: Tensor
        """Self-consistent Hamiltonian (shape: [..., orbs, orbs])"""

        density: Tensor
        """Density matrix"""

        evals: Tensor
        """
        Orbital energies, i.e., eigenvalues of Fock matrix
        (shape: [..., orbs])
        """

        evecs: Tensor
        """
        LCAO coefficients, i.e., eigenvectors of Fock matrix
        (shape: [..., orbs, orbs])
        """

        iter: int
        """Number of iterations."""

        def __init__(
            self,
            hcore: Tensor,
            overlap: Tensor,
            occupation: Tensor,
            n0: Tensor,
            numbers: Tensor,
            ihelp: IndexHelper,
            cache: InteractionList.Cache,
        ) -> None:
            self.hcore = hcore
            self.overlap = overlap
            self.occupation = occupation
            self.n0 = n0
            self.numbers = numbers
            self.ihelp = ihelp
            self.cache = cache
            self.init_zeros()

            self.iter = 1

        def init_zeros(self) -> None:
            self.energy = torch.zeros_like(self.n0)
            self.hamiltonian = torch.zeros_like(self.hcore)
            self.density = torch.zeros_like(self.hcore)
            self.evals = torch.zeros_like(self.n0)
            self.evecs = torch.zeros_like(self.hcore)

            self.old_charges = torch.zeros_like(self.energy)
            self.old_energy = torch.zeros_like(self.numbers)
            self.old_density = torch.zeros_like(self.density)

        def reset(self) -> None:
            self.iter = 0
            self.init_zeros()

        def cull(self, conv: Tensor, slicers: Slicers) -> None:
            onedim = [~conv, *slicers["orbital"]]
            twodim = [~conv, *slicers["orbital"], *slicers["orbital"]]

            self.numbers = self.numbers[[~conv, *slicers["atom"]]]
            self.overlap = self.overlap[twodim]
            self.hamiltonian = self.hamiltonian[twodim]
            self.hcore = self.hcore[twodim]
            self.occupation = self.occupation[twodim]
            self.evecs = self.evecs[twodim]
            self.evals = self.evals[onedim]
            self.n0 = self.n0[onedim]
            self.ihelp.cull(conv, slicers=slicers)
            self.cache.cull(conv, slicers=slicers)

            self.old_charges = self.old_charges[onedim]
            self.old_energy = self.old_energy[onedim]
            self.old_density = self.old_density[twodim]

    _data: _Data
    """Persistent data"""

    interactions: InteractionList
    """Interactions to minimize in self-consistent iterations"""

    fwd_options: dict[str, Any]
    """Options for forwards pass"""

    bck_options: dict[str, Any]
    """Options for backwards pass"""

    eigen_options: dict[str, Any]
    """Options for eigensolver"""

    scf_options: dict[str, Any]
    """
    Options for SCF:
    - "etemp": Electronic temperature (in a.u.) for Fermi smearing.
    - "fermi_maxiter": Maximum number of iterations for Fermi smearing.
    - "fermi_thresh": Float data type dependent threshold for Fermi iterations.
    - "fermi_fenergy_partition": Partitioning scheme for electronic free energy.
    """

    use_potential: bool
    """Whether to use the potential or the charges"""

    batched: bool
    """Whether multiple systems or a single one are handled"""

    def __init__(
        self,
        interactions: InteractionList,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.use_potential = kwargs.pop("use_potential", defaults.USE_POTENTIAL)
        self.bck_options = {"posdef": True, **kwargs.pop("bck_options", {})}
        self.fwd_options = {
            "force_convergence": False,
            "method": "broyden1",
            "alpha": -0.5,
            "f_tol": defaults.XITORCH_FATOL,
            "x_tol": defaults.XITORCH_XATOL,
            "f_rtol": float("inf"),
            "x_rtol": float("inf"),
            "maxiter": defaults.MAXITER,
            "verbose": False,
            "line_search": False,
            **kwargs.pop("fwd_options", {}),
        }

        self.eigen_options = {"method": "exacteig", **kwargs.pop("eigen_options", {})}

        self.scf_options = {**kwargs.pop("scf_options", {})}

        self._data = self._Data(*args, **kwargs)

        self.kt = self._data.hcore.new_tensor(
            self.scf_options.get("etemp", defaults.ETEMP) * K2AU
        )
        self.interactions = interactions
        self.batched = self._data.numbers.ndim > 1

    @abstractmethod
    def scf(self, guess: Tensor) -> Tensor:
        """
        Mixing and convergence for self-consistent field iterations.

        Parameters
        ----------
        guess : Tensor
            Orbital-resolved guess for charges or potential vector.

        Returns
        -------
        Tensor
            Converged, orbital-resolved charges.
        """

    def __call__(self, charges: Tensor | None = None) -> dict[str, Tensor]:
        """
        Run the self-consistent iterations until a stationary solution is reached

        Parameters
        ----------
        charges : Tensor, optional
            Initial orbital charges vector. If `None` is given (default), a
            zeros vector is used.

        Returns
        -------
        Tensor
            Converged orbital charges vector.
        """

        if charges is None:
            charges = torch.zeros_like(self._data.occupation)
        guess = self.charges_to_potential(charges) if self.use_potential else charges

        if self.scf_options["verbosity"] > 0:
            print(
                f"\n{'iter':<5} {'energy':<24} {'energy change':<15}"
                f"{'P norm change':<15} {'charge change':<15}"
            )
            print(77 * "-")

        # main SCF function (mixing)
        charges = self.scf(guess)

        if self.scf_options["verbosity"] > 0:
            print(77 * "-")

        # evaluate final energy
        energy = self.get_energy(charges)
        fenergy = self.get_electronic_free_energy()

        return {
            "charges": charges,
            "coefficients": self._data.evecs,
            "density": self._data.density,
            "emo": self._data.evals,
            "energy": energy,
            "fenergy": fenergy,
            "hamiltonian": self._data.hamiltonian,
            "occupation": self._data.occupation,
            "potential": self.charges_to_potential(charges),
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
        energy = self._data.ihelp.reduce_orbital_to_atom(self._data.energy)
        return energy + self.interactions.get_energy(
            charges, self._data.cache, self._data.ihelp
        )

    def get_electronic_free_energy(self) -> Tensor:
        r"""
        Calculate electronic free energy from entropy.

        .. math::

            G = -TS = k_B\sum_{i}f_i \; ln(f_i) + (1 - f_i)\; ln(1 - f_i))

        The atomic partitioning can be performed by means of Mulliken population
        analysis using an "electronic entropy" density matrix.

        .. math::

            E_\kappa^\text{TS} = (\mathbf P^\text{TS} \mathbf S)_{\kappa\kappa}
            \qquad\text{with}\quad \mathbf P^\text{TS} = \mathbf C^T \cdot
            \text{diag}(g) \cdot \mathbf C

        Returns
        -------
        Tensor
            Orbital-resolved electronic free energy (G = -TS).

        Note
        ----
        Partitioning scheme is set through SCF options
        (`scf_options["fermi_fenergy_partition"]`).
        Defaults to an equal partitioning to all atoms (`"equal"`).
        """

        occ = self._data.occupation
        g = torch.log(occ**occ * (1 - occ) ** (1 - occ)).sum(-2) * self.kt

        mode = self.scf_options.get(
            "fermi_fenergy_partition", defaults.FERMI_FENERGY_PARTITION
        )

        # partition to atoms equally
        if mode == "equal":
            real = real_atoms(self._data.numbers)

            count = real.count_nonzero(dim=-1).unsqueeze(-1)
            g_atomic = torch.sum(g, dim=-1, keepdim=True) / count

            return torch.where(real, g_atomic.expand(*real.shape), g.new_tensor(0.0))

        # partition to atoms via Mulliken population analysis
        if mode == "atomic":
            # "electronic entropy" density matrix
            density = torch.einsum(
                "...ik,...k,...jk->...ij",
                self._data.evecs,  # sorted by energy, starting with lowest
                g,
                self._data.evecs,  # transposed
            )

            return mulliken.get_atomic_populations(
                self._data.overlap, density, self._data.ihelp
            )

        raise ValueError(f"Unknown partitioning mode '{mode}'.")

    def iterate_charges(self, charges: Tensor) -> Tensor:
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
        if self.scf_options["verbosity"] > 0:
            if charges.ndim < 2:  # pragma: no cover
                energy = self.get_energy(charges).sum(-1).detach().clone()
                ediff = torch.linalg.vector_norm(self._data.old_energy - energy)

                density = self._data.density.detach().clone()
                pnorm = torch.linalg.matrix_norm(self._data.old_density - density)

                q = charges.detach().clone()
                qdiff = torch.linalg.vector_norm(self._data.old_charges - q)

                print(
                    f"{self._data.iter:3}   {energy: .16E}  {ediff: .6E} "
                    f"{pnorm: .6E}   {qdiff: .6E}"
                )

                self._data.old_energy = energy
                self._data.old_charges = q
                self._data.old_density = density
                self._data.iter += 1
            else:
                energy = self.get_energy(charges).detach().clone()
                ediff = torch.linalg.norm(self._data.old_energy - energy)

                density = self._data.density.detach().clone()
                pnorm = torch.linalg.norm(self._data.old_density - density)

                q = charges.detach().clone()
                qdiff = torch.linalg.norm(self._data.old_charges - q)

                print(
                    f"{self._data.iter:3}   {energy.sum(): .16E}  {ediff: .6E} "
                    f"{qdiff: .6E}"
                )

                self._data.old_energy = energy
                self._data.old_charges = q
                self._data.old_density = density
                self._data.iter += 1

        if self.fwd_options["verbose"] > 1:  # pragma: no cover
            print(f"energy: {self.get_energy(charges).sum(-1)}")
        potential = self.charges_to_potential(charges)
        return self.potential_to_charges(potential)

    def iterate_potential(self, potential: Tensor) -> Tensor:
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
        if self.scf_options["verbosity"] > 0:
            if charges.ndim < 2:  # pragma: no cover
                energy = self.get_energy(charges).sum(-1).detach().clone()
                ediff = torch.linalg.vector_norm(self._data.old_energy - energy)

                density = self._data.density.detach().clone()
                pnorm = torch.linalg.matrix_norm(self._data.old_density - density)

                q = charges.detach().clone()
                qdiff = torch.linalg.vector_norm(self._data.old_charges - q)

                print(
                    f"{self._data.iter:3}   {energy: .16E}  {ediff: .6E} "
                    f"{pnorm: .6E}   {qdiff: .6E}"
                )

                self._data.old_energy = energy
                self._data.old_charges = q
                self._data.old_density = density
                self._data.iter += 1
            else:
                energy = self.get_energy(charges).detach().clone()
                ediff = torch.linalg.norm(self._data.old_energy - energy)

                density = self._data.density.detach().clone()
                pnorm = torch.linalg.norm(self._data.old_density - density)

                q = charges.detach().clone()
                qdiff = torch.linalg.norm(self._data.old_charges - q)

                print(
                    f"{self._data.iter:3}   {energy.sum(): .16E}  {ediff: .6E} "
                    f"{qdiff: .6E}"
                )

                self._data.old_energy = energy
                self._data.old_charges = q
                self._data.old_density = density
                self._data.iter += 1

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

        return self.interactions.get_potential(
            charges, self._data.cache, self._data.ihelp
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

    def potential_to_density(self, potential: Tensor) -> Tensor:
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

    def density_to_charges(self, density: Tensor) -> Tensor:
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

        self._data.energy = torch.diagonal(
            torch.einsum("...ik,...kj->...ij", density, self._data.hcore),
            dim1=-2,
            dim2=-1,
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

    def hamiltonian_to_density(self, hamiltonian: Tensor) -> Tensor:
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

        h_op = LinearOperator.m(hamiltonian)
        o_op = self.get_overlap()
        self._data.evals, self._data.evecs = self.diagonalize(h_op, o_op)

        # round to integers to avoid numerical errors
        nel = self._data.occupation.sum(-1).round()

        # expand emo/mask to second dim (for alpha/beta electrons)
        emo = self._data.evals.unsqueeze(-2).expand([*nel.shape, -1])
        mask = self._data.ihelp.spread_shell_to_orbital(
            self._data.ihelp.orbitals_per_shell
        )
        mask = mask.unsqueeze(-2).expand([*nel.shape, -1])

        # Fermi smearing only for non-zero electronic temperature
        if self.kt is not None and not torch.all(self.kt < 3e-7):  # 0.1 Kelvin * K2AU
            self._data.occupation = filling.get_fermi_occupation(
                nel,
                emo,
                kt=self.kt,
                mask=mask,
                maxiter=self.scf_options.get("fermi_maxiter", defaults.FERMI_MAXITER),
                thr=self.scf_options.get("fermi_thresh", defaults.THRESH),
            )

            # check if number of electrons is still correct
            _nel = self._data.occupation.sum(-1)
            if torch.any(torch.abs(nel - _nel.round(decimals=3)) > 1e-4):
                raise RuntimeError(
                    f"Number of electrons changed during Fermi smearing "
                    f"({nel} -> {_nel})."
                )

        return get_density(self._data.evecs, self._data.occupation.sum(-2))

    def get_overlap(self) -> LinearOperator:
        """
        Get the overlap matrix.

        Returns
        -------
        LinearOperator
            Overlap matrix.
        """

        smat = self._data.overlap

        zeros = torch.eq(smat, 0)
        mask = torch.all(zeros, dim=-1) & torch.all(zeros, dim=-2)

        return LinearOperator.m(
            smat + torch.diag_embed(smat.new_ones(*smat.shape[:-2], 1) * mask)
        )

    def diagonalize(
        self, hamiltonian: LinearOperator, overlap: LinearOperator
    ) -> tuple[Tensor, Tensor]:
        """
        Diagonalize the Hamiltonian.

        Parameters
        ----------
        hamiltonian : LinearOperator
            Current Hamiltonian matrix.
        overlap : LinearOperator
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
    def shape(self) -> torch.Size:
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

    def getparamnames(
        self, methodname: str, prefix: str = ""
    ) -> list[str]:  # pragma: no cover
        if methodname == "scf":
            a = self.getparamnames("iterate_potential")
            b = self.getparamnames("charges_to_potential")
            c = self.getparamnames("potential_to_charges")
            return a + b + c

        if methodname == "get_energy":
            return [prefix + "_data.energy"]

        if methodname == "iterate_charges":
            a = self.getparamnames("charges_to_potential", prefix=prefix)
            b = self.getparamnames("potential_to_charges", prefix=prefix)
            return a + b

        if methodname == "iterate_potential":
            a = self.getparamnames("potential_to_charges", prefix=prefix)
            b = self.getparamnames("charges_to_potential", prefix=prefix)
            return a + b

        if methodname == "charges_to_potential":
            return []

        if methodname == "potential_to_charges":
            a = self.getparamnames("potential_to_density", prefix=prefix)
            b = self.getparamnames("density_to_charges", prefix=prefix)
            return a + b

        if methodname == "potential_to_density":
            a = self.getparamnames("potential_to_hamiltonian", prefix=prefix)
            b = self.getparamnames("hamiltonian_to_density", prefix=prefix)
            return a + b

        if methodname == "density_to_charges":
            return [
                prefix + "_data.hcore",
                prefix + "_data.overlap",
                prefix + "_data.n0",
            ]

        if methodname == "potential_to_hamiltonian":
            return [
                prefix + "_data.hcore",
                prefix + "_data.overlap",
            ]

        if methodname == "hamiltonian_to_density":
            a = [prefix + "_data.occupation"]
            b = self.getparamnames("diagonalize", prefix=prefix)
            c = self.getparamnames("get_overlap", prefix=prefix)
            return a + b + c

        if methodname == "get_overlap":
            return [prefix + "_data.overlap"]

        if methodname == "diagonalize":
            return []

        raise KeyError(f"Method '{methodname}' has no paramnames set")


def get_density(coeffs: Tensor, occ: Tensor, emo: Tensor | None = None) -> Tensor:
    """
    Calculate the density matrix from the coefficient vector and the occupation.

    Parameters
    ----------
    evecs : Tensor
        _description_
    occ : Tensor
        Occupation numbers (diagonal matrix).
    emo : Tensor | None, optional
        Orbital energies for energy weighted density matrix. Defaults to `None`.

    Returns
    -------
    Tensor
        (Energy-weighted) Density matrix.
    """
    return torch.einsum(
        "...ik,...k,...jk->...ij",
        coeffs,
        occ if emo is None else occ * emo,
        coeffs,  # transposed
    )
