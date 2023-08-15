"""
Self-consistent field
=====================
"""
from __future__ import annotations

from abc import abstractmethod

import torch

from .._types import Any, Slicers, Tensor, TypedDict
from ..basis import IndexHelper
from ..constants import K2AU, defaults
from ..exlibs.xitorch import EditableModule, LinearOperator
from ..exlibs.xitorch import linalg as xtl
from ..integral import Integrals
from ..interaction import InteractionList
from ..interaction.container import Charges, ContainerData, Potential
from ..utils import eighb, real_atoms
from ..wavefunction import filling, mulliken


class SCFResult(TypedDict):
    """Collection of SCF result variables."""

    charges: Charges
    """Self-consistent orbital-resolved Mulliken partial charges."""

    coefficients: Tensor
    """LCAO-MO coefficients (eigenvectors of Fockian)."""

    density: Tensor
    """Density matrix."""

    emo: Tensor
    """Energy of molecular orbitals (sorted by increasing energy)."""

    energy: Tensor
    """Energies of the self-consistent contributions (interactions)."""

    fenergy: Tensor
    """Atom-resolved electronic free energy from fractional occupation."""

    hamiltonian: Tensor
    """Full Hamiltonian matrix (H0 + H1)."""

    occupation: Tensor
    """Orbital occupations."""

    potential: Potential
    """Self-consistent orbital-resolved potential."""


class BaseSCF:
    """
    Self-consistent field iterator, which can be used to obtain a
    self-consistent solution for a given Hamiltonian.

    This base class lacks the `scf` method, which implements mixing and
    convergence. Additionally, the `get_overlap` and `diagonalize` method
    must be implemented.
    """

    class _Data:
        """
        Restart data for the singlepoint calculation.
        """

        numbers: Tensor
        """Atomic numbers"""

        integrals: Integrals
        """
        Collection of integrals. Core Hamiltonian and overlap are always needed.
        """

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
            occupation: Tensor,
            n0: Tensor,
            numbers: Tensor,
            ihelp: IndexHelper,
            cache: InteractionList.Cache,
            integrals: Integrals,
        ) -> None:
            if integrals.hcore is None:
                raise ValueError("No core Hamiltonian provided.")
            if integrals.overlap is None:
                raise ValueError("No Overlap provided.")

            self.ints = integrals
            self.occupation = occupation
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

            self.iter = 1

        def init_zeros(self) -> None:
            self.energy = torch.zeros_like(self.n0)
            self.hamiltonian = torch.zeros_like(self.ints.hcore)
            self.density = torch.zeros_like(self.ints.hcore)
            self.evals = torch.zeros_like(self.n0)
            self.evecs = torch.zeros_like(self.ints.hcore)

            self.old_charges = torch.zeros_like(self.energy)
            self.old_energy = torch.zeros_like(self.numbers)
            self.old_density = torch.zeros_like(self.density)

        def reset(self) -> None:
            self.iter = 0
            self.init_zeros()

        def cull(self, conv: Tensor, slicers: Slicers) -> None:
            onedim = [~conv, *slicers["orbital"]]
            twodim = [~conv, *slicers["orbital"], *slicers["orbital"]]
            threedim = [~conv, (...), *slicers["orbital"], *slicers["orbital"]]

            # disable shape check temporarily for writing culled versions back
            self.ints.run_checks = False
            self.ints.overlap = self.ints.overlap[twodim]
            self.ints.hcore = self.ints.hcore[twodim]
            if self.ints.dipole is not None:
                self.ints.dipole = self.ints.dipole[threedim]
            if self.ints.quad is not None:
                self.ints.quad = self.ints.quad[threedim]
            self.ints.run_checks = True

            self.numbers = self.numbers[[~conv, *slicers["atom"]]]
            self.hamiltonian = self.hamiltonian[twodim]
            self.occupation = self.occupation[twodim]
            self.evecs = self.evecs[twodim]
            self.evals = self.evals[onedim]
            self.energy = self.energy[onedim]
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
        self.scp_mode = self.scf_options.get("scp_mode", defaults.SCP_MODE)
        if self.scp_mode in ("charge", "charges"):
            self._fcn = self.iterate_charges
        elif self.scp_mode == "potential":
            self._fcn = self.iterate_potential
        elif self.scp_mode == "fock":
            self._fcn = self.iterate_fockian
        else:
            raise ValueError(
                f"Unknown convergence target (SCP mode) '{self.scp_mode}'."
            )

        self._data = self._Data(*args, **kwargs)

        self.kt = self._data.ints.hcore.new_tensor(
            self.scf_options.get("etemp", defaults.ETEMP) * K2AU
        )

        self.interactions = interactions
        self.batched = self._data.numbers.ndim > 1

    @abstractmethod
    def scf(self, guess: Tensor) -> Charges:
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

    @abstractmethod
    def get_overlap(self) -> LinearOperator | Tensor:
        """
        Get the overlap matrix.

        Returns
        -------
        LinearOperator | Tensor
            Overlap matrix.
        """

    @abstractmethod
    def diagonalize(self, hamiltonian: Tensor) -> tuple[Tensor, Tensor]:
        """
        Diagonalize the Hamiltonian.

        The overlap matrix is retrieved within this method using the
        `get_overlap` method.

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

    def __call__(self, charges: Tensor | Charges | None = None) -> SCFResult:
        """
        Run the self-consistent iterations until a stationary solution is
        reached.

        Parameters
        ----------
        charges : Tensor | Charges, optional
            Initial orbital charges vector. If `None` is given (default), a
            zero vector is used.

        Returns
        -------
        Tensor
            Converged orbital charges vector.
        """

        # initialize zero charges (equivalent to SAD guess)
        if charges is None:
            charges = torch.zeros_like(self._data.occupation)

        # initialize Charge container depending on given integrals
        if isinstance(charges, Tensor):
            charges = Charges(mono=charges, batched=self.batched)
            self._data.charges["mono"] = charges.mono_shape

            if self._data.ints.dipole is not None:
                shp = (*self._data.numbers.shape, defaults.DP_SHAPE)
                zeros = torch.zeros(shp, device=self.device, dtype=self.dtype)
                charges.dipole = zeros
                self._data.charges["dipole"] = charges.dipole_shape

            if self._data.ints.quad is not None:
                shp = (*self._data.numbers.shape, defaults.QP_SHAPE)
                zeros = torch.zeros(shp, device=self.device, dtype=self.dtype)
                charges.quad = zeros
                self._data.charges["quad"] = charges.quad_shape

        if self.scp_mode in ("charge", "charges"):
            guess = charges.as_tensor()
        elif self.scp_mode == "potential":
            potential = self.charges_to_potential(charges)
            guess = potential.as_tensor()
        elif self.scp_mode == "fock":
            potential = self.charges_to_potential(charges)
            guess = self.potential_to_hamiltonian(potential)
        else:
            raise ValueError(
                f"Unknown convergence target (SCP mode) '{self.scp_mode}'."
            )

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
        charges.nullify_padding()
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

    def converged_to_charges(self, x: Tensor) -> Charges:
        """
        Convert the converged property to charges.

        Parameters
        ----------
        x : Tensor
            Converged property (scp).

        Returns
        -------
        Tensor
            Orbital-resolved partial charges

        Raises
        ------
        ValueError
            Unknown `scp_mode` given.
        """

        if self.scp_mode in ("charge", "charges"):
            return Charges.from_tensor(x, self._data.charges, batched=self.batched)

        if self.scp_mode == "potential":
            pot = Potential.from_tensor(x, self._data.potential, batched=self.batched)
            return self.potential_to_charges(pot)

        if self.scp_mode == "fock":
            zero = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            x = torch.where(x != defaults.PADNZ, x, zero)

            self._data.density = self.hamiltonian_to_density(x)
            return self.density_to_charges(self._data.density)

        raise ValueError(f"Unknown convergence target (SCP mode) '{self.scp_mode}'.")

    def get_energy(self, charges: Charges) -> Tensor:
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

    def get_energy_as_dict(self, charges: Charges) -> dict[str, Tensor]:
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
        energy_h0 = {"h0": self._data.energy}

        energy_interactions = self.interactions.get_energy_as_dict(
            charges, self._data.cache, self._data.ihelp
        )

        return {**energy_h0, **energy_interactions}

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
        eps = torch.tensor(
            torch.finfo(self._data.occupation.dtype).eps,
            device=self.device,
            dtype=self.dtype,
        )

        occ = torch.clamp(self._data.occupation, min=eps)
        occ1 = torch.clamp(1 - self._data.occupation, min=eps)
        g = torch.log(occ**occ * occ1**occ1).sum(-2) * self.kt

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
                self._data.ints.overlap, density, self._data.ihelp
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

        q = Charges.from_tensor(charges, self._data.charges, batched=self.batched)
        potential = self.charges_to_potential(q)

        if self.scf_options.get("verbosity", defaults.VERBOSITY) > 0:
            if charges.ndim < 2:  # pragma: no cover
                energy = self.get_energy(q).sum(-1).detach().clone()
                ediff = torch.linalg.vector_norm(self._data.old_energy - energy)

                density = self._data.density.detach().clone()
                pnorm = torch.linalg.matrix_norm(self._data.old_density - density)

                _q = charges.detach().clone()
                qdiff = torch.linalg.vector_norm(self._data.old_charges - _q)

                print(
                    f"{self._data.iter:3}   {energy: .16E}  {ediff: .6E} "
                    f"{pnorm: .6E}   {qdiff: .6E}"
                )

                self._data.old_energy = energy
                self._data.old_charges = _q
                self._data.old_density = density
                self._data.iter += 1
            else:
                energy = self.get_energy(q).detach().clone()
                ediff = torch.linalg.norm(self._data.old_energy - energy)

                density = self._data.density.detach().clone()
                pnorm = torch.linalg.norm(self._data.old_density - density)

                _q = charges.detach().clone()
                qdiff = torch.linalg.norm(self._data.old_charges - _q)

                print(
                    f"{self._data.iter:3}   {energy.sum(): .16E}  {ediff: .6E} "
                    f"{qdiff: .6E}"
                )

                self._data.old_energy = energy
                self._data.old_charges = _q
                self._data.old_density = density
                self._data.iter += 1

        if self.fwd_options["verbose"] > 1:  # pragma: no cover
            print(f"energy: {self.get_energy(q).sum(-1)}")

        new_charges = self.potential_to_charges(potential)
        return new_charges.as_tensor()

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
        pot = Potential.from_tensor(
            potential, self._data.potential, batched=self.batched
        )
        charges = self.potential_to_charges(pot)

        if self.scf_options["verbosity"] > 0:
            if charges.mono.ndim < 2:  # pragma: no cover
                energy = self.get_energy(charges).sum(-1).detach().clone()
                ediff = torch.linalg.vector_norm(self._data.old_energy - energy)

                density = self._data.density.detach().clone()
                pnorm = torch.linalg.matrix_norm(self._data.old_density - density)

                q = charges.mono.detach().clone()
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

                q = charges.mono.detach().clone()
                qdiff = torch.linalg.norm(self._data.old_charges - q)

                print(
                    f"{self._data.iter:3}   {energy.sum(): .16E}  {ediff: .6E} "
                    f"{qdiff: .6E}"
                )

                self._data.old_energy = energy
                self._data.old_charges = q
                self._data.old_density = density
                self._data.iter += 1

        new_potential = self.charges_to_potential(charges)
        return new_potential.as_tensor()

    def iterate_fockian(self, fockian: Tensor) -> Tensor:
        """
        Perform single self-consistent iteration using the Fock matrix.

        Parameters
        ----------
        fockian : Tensor
            Fock matrix.

        Returns
        -------
        Tensor
            New Fock matrix.
        """
        self._data.density = self.hamiltonian_to_density(fockian)
        charges = self.density_to_charges(self._data.density)
        potential = self.charges_to_potential(charges)
        self._data.hamiltonian = self.potential_to_hamiltonian(potential)

        return self._data.hamiltonian

    def charges_to_potential(self, charges: Charges) -> Potential:
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

        potential = self.interactions.get_potential(
            charges, self._data.cache, self._data.ihelp
        )
        self._data.potential = {
            "mono": potential.mono_shape,
            "dipole": potential.dipole_shape,
            "quad": potential.quad_shape,
            "label": potential.label,
        }

        return potential

    def potential_to_charges(self, potential: Potential) -> Charges:
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

    def potential_to_density(self, potential: Potential) -> Tensor:
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

    def density_to_charges(self, density: Tensor) -> Charges:
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

        ints = self._data.ints

        # Calculate diagonal directly by using index "i" twice on left side.
        # The slower but more readable approach would instead compute the full
        # matrix with "...ik,...kj->...ij" and only extract the diagonal
        # afterwards with `torch.diagonal(tensor, dim1=-2, dim2=-1)`.
        self._data.energy = torch.einsum("...ik,...ki->...i", density, ints.hcore)

        # monopolar charges
        populations = torch.einsum("...ik,...ki->...i", density, ints.overlap)
        charges = Charges(mono=self._data.n0 - populations, batched=self.batched)

        # Atomic dipole moments (dipole charges)
        if ints.dipole is not None:
            # Again, the diagonal is directly calculated instead of full matrix
            # ("...ik,...mkj->...ijm") as `torch.diagonal` behaves weirdly for
            # more than 2D tensors. Additionally, we move the multipole
            # dimension to the back, which is required for the reduction to
            # atom-resolution.
            charges.dipole = self._data.ihelp.reduce_orbital_to_atom(
                -torch.einsum("...ik,...mki->...im", density, ints.dipole),
                extra=True,
                dim=-2,
            )

        # Atomic quadrupole moments (quadrupole charges)
        if ints.quad is not None:
            charges.quad = self._data.ihelp.reduce_orbital_to_atom(
                -torch.einsum("...ik,...mki->...im", density, ints.quad),
                extra=True,
                dim=-2,
            )

        return charges

    def potential_to_hamiltonian(self, potential: Potential) -> Tensor:
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

        h1 = self._data.ints.hcore

        if potential.mono is not None:
            v = potential.mono.unsqueeze(-1) + potential.mono.unsqueeze(-2)
            h1 = h1 - (0.5 * self._data.ints.overlap * v)

        def add_vmp_to_h1(h1: Tensor, mpint: Tensor, vmp: Tensor) -> Tensor:
            # TODO: Place better?
            from dxtb.utils.batch import index

            # spread potential to orbitals
            v = index(vmp, self._data.ihelp.orbitals_to_atom)

            # Form dot product over the the multipolar components.
            #  - shape multipole integral: (..., x, norb, norb)
            #  - shape multipole potential: (..., norb, x)
            tmp = 0.5 * torch.einsum("...kij,...ik->...ij", mpint, v)
            return h1 - (tmp + tmp.mT)

        if potential.dipole is not None:
            dpint = self._data.ints.dipole
            if dpint is not None:
                h1 = add_vmp_to_h1(h1, dpint, potential.dipole)

        # print("potential.dipole", potential.dipole)
        # print("potential.quad", potential.quad)
        if potential.quad is not None:
            qpint = self._data.ints.quad
            if qpint is not None:
                h1 = add_vmp_to_h1(h1, qpint, potential.quad)

        return h1

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

        self._data.evals, self._data.evecs = self.diagonalize(hamiltonian)

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

    @property
    def shape(self) -> torch.Size:
        """
        Returns the shape of the density matrix in this engine.
        """
        return self._data.ints.hcore.shape

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the tensors in this engine.
        """
        return self._data.ints.hcore.dtype

    @property
    def device(self) -> torch.device:
        """
        Returns the device of the tensors in this engine.
        """
        return self._data.ints.hcore.device


class BaseXSCF(BaseSCF, EditableModule):
    """
    Base class for the `xitorch`-based self-consistent field iterator.

    This base class implements the `get_overlap` and the `diagonalize` methods
    that use `LinearOperator`s. Additionally, `getparamnames` is implemented,
    which is mandatory for all descendents of `xitorch`s base class called
    `EditableModule`.

    This class only lacks the `scf` method, which implements mixing and
    convergence.
    """

    def get_overlap(self) -> LinearOperator:
        """
        Get the overlap matrix.

        Returns
        -------
        LinearOperator
            Overlap matrix.
        """

        smat = self._data.ints.overlap

        zeros = torch.eq(smat, 0)
        mask = torch.all(zeros, dim=-1) & torch.all(zeros, dim=-2)

        return LinearOperator.m(
            smat + torch.diag_embed(smat.new_ones(*smat.shape[:-2], 1) * mask)
        )

    def diagonalize(self, hamiltonian: Tensor) -> tuple[Tensor, Tensor]:
        """
        Diagonalize the Hamiltonian.

        The overlap matrix is retrieved within this method using the
        `get_overlap` method.

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

        h_op = LinearOperator.m(hamiltonian)
        o_op = self.get_overlap()

        return xtl.lsymeig(A=h_op, M=o_op, **self.eigen_options)

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

        if methodname == "iterate_fockian":
            a = self.getparamnames("hamiltonian_to_density", prefix=prefix)
            b = self.getparamnames("density_to_charges", prefix=prefix)
            c = self.getparamnames("charges_to_potential", prefix=prefix)
            d = self.getparamnames("potential_to_hamiltonian", prefix=prefix)
            return a + b + c + d

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
                prefix + "_data.ints.hcore",
                prefix + "_data.ints.overlap",
                prefix + "_data.n0",
            ]

        if methodname == "potential_to_hamiltonian":
            return [
                prefix + "_data.ints.hcore",
                prefix + "_data.ints.overlap",
            ]

        if methodname == "hamiltonian_to_density":
            a = [prefix + "_data.occupation"]
            b = self.getparamnames("diagonalize", prefix=prefix)
            c = self.getparamnames("get_overlap", prefix=prefix)
            return a + b + c

        if methodname == "get_overlap":
            return [prefix + "_data.ints.overlap"]

        if methodname == "diagonalize":
            return []

        raise KeyError(f"Method '{methodname}' has no paramnames set")


class BaseTSCF(BaseSCF):
    """
    Base class for a standard self-consistent field iterator.

    This base class implements the `get_overlap` and the `diagonalize` methods
    using plain tensors. The diagonalization routine is taken from TBMaLT
    (hence the T in the class name).

    This base class only lacks the `scf` method, which implements mixing and
    convergence.
    """

    def get_overlap(self) -> Tensor:
        """
        Get the overlap matrix.

        Returns
        -------
        Tensor
            Overlap matrix.
        """

        smat = self._data.ints.overlap

        zeros = torch.eq(smat, 0)
        mask = torch.all(zeros, dim=-1) & torch.all(zeros, dim=-2)

        return smat + torch.diag_embed(smat.new_ones(*smat.shape[:-2], 1) * mask)

    def diagonalize(self, hamiltonian: Tensor) -> tuple[Tensor, Tensor]:
        """
        Diagonalize the Hamiltonian.

        The overlap matrix is retrieved within this method using the
        `get_overlap` method.

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

        o = self.get_overlap()
        return eighb(
            a=hamiltonian,
            b=o,
            is_posdef=True,
            factor=torch.finfo(self.dtype).eps ** 0.5,
        )


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
