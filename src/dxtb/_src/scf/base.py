# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Self-consistent field
=====================
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch
from tad_mctc.batch import real_atoms
from tad_mctc.math import einsum
from tad_mctc.units import KELVIN2AU

from dxtb import IndexHelper, OutputHandler
from dxtb._src.components.interactions.container import (
    Charges,
    ContainerData,
    Potential,
)
from dxtb._src.constants import defaults, labels
from dxtb._src.timing.decorator import timer_decorator
from dxtb._src.typing import DD, Any, Literal, Slicers, Tensor, overload
from dxtb._src.wavefunction import filling, mulliken
from dxtb.config import ConfigSCF

from .result import SCFResult
from .utils import get_density

if TYPE_CHECKING:
    from dxtb._src.components.interactions import InteractionList, InteractionListCache
    from dxtb._src.exlibs import xitorch as xt
    from dxtb._src.integral.container import IntegralMatrices
del TYPE_CHECKING

__all__ = ["BaseSCF"]


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

        integrals: IntegralMatrices
        """
        Collection of integrals. Core Hamiltonian and overlap are always needed.
        """

        occupation: Tensor
        """Occupation numbers (shape: [..., 2, orbs])"""

        n0: Tensor
        """Reference occupation for each orbital (shape: [..., orbs])"""

        ihelp: IndexHelper
        """Index mapping for the basis set"""

        cache: InteractionListCache
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
            cache: InteractionListCache,
            integrals: IntegralMatrices,
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

            self.iter = -1  # bumped before printing, guess energy also printed

        def init_zeros(self) -> None:
            """Initialize all tensors with zeros."""
            self.energy = torch.zeros_like(self.n0)
            self.hamiltonian = torch.zeros_like(self.ints.hcore)
            self.density = torch.zeros_like(self.ints.hcore)
            self.evals = torch.zeros_like(self.n0)
            self.evecs = torch.zeros_like(self.ints.hcore)

            self.old_charges = torch.zeros_like(self.energy)
            self.old_energy = torch.zeros_like(self.numbers)
            self.old_density = torch.zeros_like(self.density)

        def reset(self) -> None:
            """Reset all tensors and iteration count to zero."""
            self.iter = 0
            self.init_zeros()

        def cull(self, conv: Tensor, slicers: Slicers) -> None:
            """
            Cull all tensors according to the given slicers.

            Parameters
            ----------
            conv : Tensor
                Convergence mask.
            slicers : Slicers
                Slicers for the tensors.
            """
            onedim = [~conv, *slicers["orbital"]]
            onedim_atom = [~conv, *slicers["atom"]]
            twodim = [~conv, *slicers["orbital"], *slicers["orbital"]]
            threedim = [~conv, (...), *slicers["orbital"], *slicers["orbital"]]

            # disable shape check temporarily for writing culled versions back
            self.ints.run_checks = False
            self.ints.overlap = self.ints.overlap[twodim]
            self.ints.hcore = self.ints.hcore[twodim]
            if self.ints.dipole is not None:
                self.ints.dipole = self.ints.dipole[threedim]
            if self.ints.quadrupole is not None:
                self.ints.quadrupole = self.ints.quadrupole[threedim]
            self.ints.run_checks = True

            self.numbers = self.numbers[[~conv, *slicers["atom"]]]
            self.hamiltonian = self.hamiltonian[twodim]
            self.density = self.density[twodim]
            self.occupation = self.occupation[twodim]
            self.evecs = self.evecs[twodim]
            self.evals = self.evals[onedim]
            self.energy = self.energy[onedim]
            self.n0 = self.n0[onedim]
            self.ihelp.cull(conv, slicers=slicers)
            self.cache.cull(conv, slicers=slicers)

            self.old_charges = self.old_charges[onedim]
            self.old_energy = self.old_energy[onedim_atom]
            self.old_density = self.old_density[twodim]

    _data: _Data
    """Persistent data"""

    config: ConfigSCF
    """Configuration object for the SCF procedure."""

    interactions: InteractionList
    """Interactions to minimize in self-consistent iterations"""

    fwd_options: dict[str, Any]
    """Options for forwards pass"""

    bck_options: dict[str, Any]
    """Options for backwards pass"""

    eigen_options: dict[str, Any]
    """Options for eigensolver"""

    batch_mode: int
    """Whether multiple systems or a single one are handled"""

    def __init__(
        self,
        interactions: InteractionList,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if "config" in kwargs:
            self.config = kwargs.pop("config")
            if not isinstance(self.config, ConfigSCF):
                raise ValueError("Invalid configuration object.")
        else:
            self.config = ConfigSCF()

        # TODO: Move these settings to config
        self.bck_options = {"posdef": True, **kwargs.pop("bck_options", {})}
        self.fwd_options = {
            "force_convergence": False,
            "method": "broyden1",
            "alpha": -0.5,
            "damp": self.config.damp,
            "f_tol": self.config.f_atol,
            "x_tol": self.config.x_atol,
            "f_rtol": float("inf"),
            "x_rtol": float("inf"),
            "maxiter": self.config.maxiter,
            "verbose": False,
            "line_search": False,
            **kwargs.pop("fwd_options", {}),
        }

        self.eigen_options = {"method": "exacteig", **kwargs.pop("eigen_options", {})}

        if self.config.scp_mode == labels.SCP_MODE_CHARGE:
            self._fcn = self.iterate_charges
        elif self.config.scp_mode == labels.SCP_MODE_POTENTIAL:
            self._fcn = self.iterate_potential
        elif self.config.scp_mode == labels.SCP_MODE_FOCK:
            self._fcn = self.iterate_fockian
        else:
            raise ValueError(
                f"Unknown convergence target (SCP mode) '{self.config.scp_mode}'."
            )

        self._data = self._Data(*args, **kwargs)

        self.kt = torch.tensor(self.config.fermi.etemp * KELVIN2AU, **self.dd)

        self.interactions = interactions

    @overload
    @abstractmethod
    def scf(
        self,
        guess: Tensor,
        return_charges: Literal[True] = True,
    ) -> Charges: ...

    @overload
    @abstractmethod
    def scf(
        self,
        guess: Tensor,
        return_charges: Literal[False] = False,
    ) -> Charges | Potential | Tensor: ...

    @abstractmethod
    def scf(
        self, guess: Tensor, return_charges: bool = True
    ) -> Charges | Potential | Tensor:
        """
        Mixing and convergence for self-consistent field iterations.

        Parameters
        ----------
        guess : Tensor
            Orbital-resolved guess for charges, potential or Fock matrix.
        return_charges : bool, optional
            Whether to return the charges. Default is ``True``.

        Returns
        -------
        Tensor
            Converged, orbital-resolved charges.
        """

    @abstractmethod
    def get_overlap(self) -> xt.LinearOperator | Tensor:
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

    def _guess(self, charges: Tensor | Charges | None) -> Tensor:
        """
        Get the initial guess for the charges depending on the convergence
        property (self-consistent property).

        Parameters
        ----------
        charges : Tensor | Charges, optional
            Initial orbital charges vector. If ``None`` is given (default), a
            zero vector is used.

        Returns
        -------
        Tensor
            Initial guess for the charges.
        """

        # initialize zero charges (equivalent to SAD guess)
        if charges is None:
            charges = torch.zeros_like(self._data.occupation)

        # initialize Charge container depending on given integrals
        if isinstance(charges, Tensor):
            charges = Charges(mono=charges, batch_mode=self.config.batch_mode)
            self._data.charges["mono"] = charges.mono_shape

            if self._data.ints.dipole is not None:
                shp = (*self._data.numbers.shape, defaults.DP_SHAPE)
                zeros = torch.zeros(shp, **self.dd)
                charges.dipole = zeros
                self._data.charges["dipole"] = charges.dipole_shape

            if self._data.ints.quadrupole is not None:
                shp = (*self._data.numbers.shape, defaults.QP_SHAPE)
                zeros = torch.zeros(shp, **self.dd)
                charges.quad = zeros
                self._data.charges["quad"] = charges.quad_shape

        if self.config.scp_mode == labels.SCP_MODE_CHARGE:
            return charges.as_tensor()

        if self.config.scp_mode == labels.SCP_MODE_POTENTIAL:
            potential = self.charges_to_potential(charges)
            return potential.as_tensor()

        if self.config.scp_mode == labels.SCP_MODE_FOCK:
            potential = self.charges_to_potential(charges)
            return self.potential_to_hamiltonian(potential)

        lbls = (
            labels.SCP_MODE_POTENTIAL_STRS
            + labels.SCP_MODE_CHARGE_STRS
            + labels.SCP_MODE_FOCK_STRS
        )
        raise ValueError(
            "Unknown convergence target (SCP mode) '"
            f"{self.config.scp_mode}'. Use one of {', '.join(lbls)}."
        )

    def __call__(self, charges: Tensor | Charges | None = None) -> SCFResult:
        """
        Run the self-consistent iterations until a stationary solution is
        reached.

        Parameters
        ----------
        charges : Tensor | Charges, optional
            Initial orbital charges vector. If ``None`` is given (default), a
            zero vector is used.

        Returns
        -------
        Tensor
            Converged orbital charges vector.
        """

        guess = self._guess(charges)

        # main SCF function (mixing)
        OutputHandler.write_stdout(
            f"\n{'iter':<5} {'Energy':<24} {'Delta E':<16}"
            f"{'Delta Pnorm':<15} {'Delta q':<15}",
            v=3,
        )
        OutputHandler.write_stdout(77 * "-", v=3)

        q = self.scf(guess)

        OutputHandler.write_stdout(77 * "-", v=3)
        OutputHandler.write_stdout("", v=3)

        # evaluate final energy
        q.nullify_padding()
        energy = self.get_energy(q)
        fenergy = self.get_electronic_free_energy()

        return {
            "charges": q,
            "coefficients": self._data.evecs,
            "density": self._data.density,
            "emo": self._data.evals,
            "energy": energy,
            "fenergy": fenergy,
            "hamiltonian": self._data.hamiltonian,
            "occupation": self._data.occupation,
            "potential": self.charges_to_potential(q),
            "iterations": self._data.iter,
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

        if self.config.scp_mode == labels.SCP_MODE_CHARGE:
            return Charges.from_tensor(
                x, self._data.charges, batch_mode=self.config.batch_mode
            )

        if self.config.scp_mode == labels.SCP_MODE_POTENTIAL:
            pot = Potential.from_tensor(
                x, self._data.potential, batch_mode=self.config.batch_mode
            )
            return self.potential_to_charges(pot)

        if self.config.scp_mode == labels.SCP_MODE_FOCK:
            zero = torch.tensor(0.0, **self.dd)
            x = torch.where(x != defaults.PADNZ, x, zero)

            self._data.density = self.hamiltonian_to_density(x)
            return self.density_to_charges(self._data.density)

        raise ValueError(
            f"Unknown convergence target (SCP mode) '{self.config.scp_mode}'."
        )

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
        Partitioning scheme is set through SCF config
        (`config.fermi.partition`).
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

        mode = self.config.fermi.partition

        # partition to atoms equally
        if mode == labels.FERMI_PARTITION_EQUAL:
            real = real_atoms(self._data.numbers)

            count = real.count_nonzero(dim=-1).unsqueeze(-1)
            g_atomic = torch.sum(g, dim=-1, keepdim=True) / count

            return torch.where(
                real, g_atomic.expand(*real.shape), torch.tensor(0.0, **self.dd)
            )

        # partition to atoms via Mulliken population analysis
        if mode == labels.FERMI_PARTITION_ATOMIC:
            # "electronic entropy" density matrix
            density = einsum(
                "...ik,...k,...jk->...ij",
                self._data.evecs,  # sorted by energy, starting with lowest
                g,
                self._data.evecs,  # transposed
            )

            return mulliken.get_atomic_populations(
                self._data.ints.overlap, density, self._data.ihelp
            )

        raise ValueError(f"Unknown partitioning mode '{mode}'.")

    def _print(self, charges: Charges) -> None:
        self._data.iter += 1

        # explicitly check to avoid some superfluos calculations
        if OutputHandler.verbosity < 3:
            return

        if charges.mono.ndim < 2:  # pragma: no cover
            energy = self.get_energy(charges).sum(-1).detach().clone()
            ediff = (
                (self._data.old_energy.sum(-1) - energy) if self._data.iter > 0 else 0.0
            )

            density = self._data.density.detach().clone()
            pnorm = (
                torch.linalg.matrix_norm(self._data.old_density - density)
                if self._data.iter > 0
                else 0.0
            )

            _q = charges.mono.detach().clone()
            qdiff = (
                torch.linalg.vector_norm(self._data.old_charges - _q)
                if self._data.iter > 0
                else 0.0
            )

            OutputHandler.write_row(
                "SCF Iterations",
                f"{self._data.iter:3}",
                [
                    f"{energy: .14E}",
                    f"{ediff: .6E}",
                    f"{pnorm: .6E}",
                    f"{qdiff: .6E}",
                ],
            )

            self._data.old_energy = energy
            self._data.old_charges = _q
            self._data.old_density = density
        else:
            energy = self.get_energy(charges).detach().clone()
            ediff = (
                torch.linalg.norm(self._data.old_energy - energy)
                if self._data.iter > 0
                else 0.0
            )

            density = self._data.density.detach().clone()
            pnorm = (
                torch.linalg.norm(self._data.old_density - density)
                if self._data.iter > 0
                else 0.0
            )

            _q = charges.mono.detach().clone()
            qdiff = (
                torch.linalg.norm(self._data.old_charges - _q)
                if self._data.iter > 0
                else 0.0
            )

            OutputHandler.write_row(
                "SCF Iterations",
                f"{self._data.iter:3}",
                [
                    f"{energy.norm(): .14E}",
                    f"{ediff: .6E}",
                    f"{pnorm: .6E}",
                    f"{qdiff: .6E}",
                ],
            )

            self._data.old_energy = energy
            self._data.old_charges = _q
            self._data.old_density = density

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

        q = Charges.from_tensor(
            charges, self._data.charges, batch_mode=self.config.batch_mode
        )
        potential = self.charges_to_potential(q)

        # FIXME: Batch print not working!
        self._print(q)

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
            potential, self._data.potential, batch_mode=self.config.batch_mode
        )
        charges = self.potential_to_charges(pot)

        # FIXME: Batch print not working!
        self._print(charges)

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

        # FIXME: Batch print not working!
        self._print(charges)

        return self._data.hamiltonian

    @timer_decorator("Potential", "SCF")
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

    @timer_decorator("Charges", "SCF")
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
        self._data.energy = einsum("...ik,...ki->...i", density, ints.hcore)

        # monopolar charges
        populations = einsum("...ik,...ki->...i", density, ints.overlap)
        charges = Charges(
            mono=self._data.n0 - populations, batch_mode=self.config.batch_mode
        )

        # Atomic dipole moments (dipole charges)
        if ints.dipole is not None:
            # Again, the diagonal is directly calculated instead of full matrix
            # ("...ik,...mkj->...ijm") as `torch.diagonal` behaves weirdly for
            # more than 2D tensors. Additionally, we move the multipole
            # dimension to the back, which is required for the reduction to
            # atom-resolution.
            charges.dipole = self._data.ihelp.reduce_orbital_to_atom(
                -einsum("...ik,...mki->...im", density, ints.dipole),
                extra=True,
                dim=-2,
            )

        # Atomic quadrupole moments (quadrupole charges)
        if ints.quadrupole is not None:
            charges.quad = self._data.ihelp.reduce_orbital_to_atom(
                -einsum("...ik,...mki->...im", density, ints.quadrupole),
                extra=True,
                dim=-2,
            )

        return charges

    @timer_decorator("Fock build", "SCF")
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
            # spread potential to orbitals
            v = self._data.ihelp.spread_atom_to_orbital(vmp, dim=-2, extra=True)

            # Form dot product over the the multipolar components.
            #  - shape multipole integral: (..., x, norb, norb)
            #  - shape multipole potential: (..., norb, x)
            tmp = 0.5 * einsum("...kij,...ik->...ij", mpint, v)
            return h1 - (tmp + tmp.mT)

        if potential.dipole is not None:
            dpint = self._data.ints.dipole
            if dpint is not None:
                h1 = add_vmp_to_h1(h1, dpint, potential.dipole)

        if potential.quad is not None:
            qpint = self._data.ints.quadrupole
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
                maxiter=self.config.fermi.maxiter,
                thr=self.config.fermi.thresh,
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

    @property
    def dd(self) -> DD:
        """
        Returns the device of the tensors in this engine.
        """
        return {"device": self.device, "dtype": self.dtype}
