"""
Base calculator for the extended tight-binding model.
"""
from __future__ import annotations

import warnings
from functools import wraps

import torch

from .. import ncoord, scf
from .._types import Any, Callable, Sequence, Tensor, TensorLike
from ..basis import Basis, IndexHelper
from ..classical import (
    Classical,
    ClassicalList,
    Halogen,
    Repulsion,
    new_halogen,
    new_repulsion,
)
from ..constants import defaults
from ..coulomb import new_es2, new_es3
from ..data import cov_rad_d3
from ..dispersion import Dispersion, new_dispersion
from ..integral import Integrals, Overlap, OverlapLibcint
from ..integral import libcint as intor
from ..interaction import Charges, Interaction, InteractionList, Potential
from ..interaction.external import field as efield
from ..param import Param, get_elem_angular
from ..utils import Timers, ToleranceWarning, batch
from ..utils.misc import is_basis_list
from ..wavefunction import filling
from ..xtb.h0 import Hamiltonian
from .h0 import Hamiltonian

__all__ = ["Calculator", "Result"]


def use_intdriver(driver_arg: int = 1) -> Callable:
    def decorator(fcn: Callable) -> Callable:
        @wraps(fcn)
        def wrap(self: Calculator, *args: Any, **kwargs: Any) -> Any:
            if self.opts["int_driver"] == "libcint":
                self.init_intdriver(args[driver_arg])

            result = fcn(self, *args, **kwargs)
            return result

        return wrap

    return decorator


class Result(TensorLike):
    """
    Result container for singlepoint calculation.
    """

    charges: Charges
    """Self-consistent orbital-resolved Mulliken partial charges."""

    coefficients: Tensor
    """LCAO-MO coefficients (eigenvectors of Fockian)."""

    density: Tensor
    """Density matrix."""

    cenergies: dict[str, Tensor]
    """Energies of classical contributions."""

    cgradients: dict[str, Tensor]
    """Gradients of classical contributions."""

    emo: Tensor
    """Energy of molecular orbitals (sorted by increasing energy)."""

    fenergy: Tensor
    """Atom-resolved electronic free energy from fractional occupation."""

    gradient: Tensor | None
    """Gradient of total energy w.r.t. positions"""

    hamiltonian: Tensor
    """Full Hamiltonian matrix (H0 + H1)."""

    hamiltonian_grad: Tensor
    """Nuclear gradient of Hamiltonian matrix."""

    integrals: Integrals
    """Collection of integrals including overlap and core Hamiltonian (H0)."""

    interaction_grad: Tensor
    """Nuclear gradient of interactions"""

    occupation: Tensor
    """Orbital occupations."""

    overlap_grad: Tensor
    """Nuclear gradient of overlap matrix."""

    potential: Potential
    """Self-consistent potentials."""

    scf: Tensor
    """Atom-resolved energy from the self-consistent field (SCF) calculation."""

    timer: Timers | None
    """Collection of timers for all steps."""

    total: Tensor
    """Total energy."""

    total_grad: Tensor
    """Total nuclear gradient."""

    __slots__ = [
        "charges",
        "coefficients",
        "cenergies",
        "cgradients",
        "density",
        "emo",
        "fenergy",
        "gradient",
        "hamiltonian",
        "hamiltonian_grad",
        "integrals",
        "interaction_grad",
        "occupation",
        "overlap_grad",
        "potential",
        "scf",
        "timer",
        "total",
        "total_grad",
    ]

    def __init__(
        self,
        positions: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)
        shape = positions.shape[:-1]

        self.gradient = None
        self.timer = None
        self.scf = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.fenergy = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.hamiltonian_grad = torch.zeros_like(positions)
        self.interaction_grad = torch.zeros_like(positions)
        self.overlap_grad = torch.zeros_like(positions)
        self.total = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.total_grad = torch.zeros_like(positions)

    def __repr__(self) -> str:  # pragma: no cover
        """Custom print representation showing all available slots."""
        return f"{self.__class__.__name__}({self.__slots__})"

    def print_energies(self, name: str = "Energy", width: int = 50) -> None:
        """Print energies in a table."""

        labels = {
            "cenergies": "Classical contribution energies",
            "fenergy": "Electronic free energy",
            "scf": "Electronic Energy (SCF)",
        }

        print(f"{name:*^50}\n")
        print("{:<27}  {:<18}".format("Contribution", "Energy in a.u."))
        print(width * "-")

        tot = "Total Energy"
        total = torch.sum(self.total, dim=-1)

        for label, n in labels.items():
            if not hasattr(self, label):
                continue
            energy = getattr(self, label)
            if isinstance(energy, dict):
                for key, value in energy.items():
                    e = torch.sum(value, dim=-1)
                    print(f"{key:<27} {e: .16f}")
                continue

            e = torch.sum(getattr(self, label), dim=-1)
            print(f"{n:<27} {e: .16f}")

        print(width * "-")
        print(f"{tot:<27} {total: .16f}")
        print("")


class Calculator(TensorLike):
    """
    Parametrized calculator defining the extended tight-binding model.

    The calculator holds the atomic orbital basis set for defining the Hamiltonian
    and the overlap matrix.
    """

    hamiltonian: Hamiltonian
    """Core Hamiltonian definition."""

    dispersion: Dispersion | None = None
    """Dispersion definition."""

    repulsion: Repulsion | None = None
    """Repulsion definition."""

    halogen: Halogen | None = None
    """Halogen bond definition."""

    interactions: InteractionList
    """Interactions to minimize in self-consistent iterations."""

    classicals: ClassicalList
    """Classical contributions."""

    ihelp: IndexHelper
    """Helper class for indexing."""

    opts: dict[str, Any]
    """Calculator options."""

    timer: Timers
    """Collection of timers."""

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        *,
        classical: Sequence[Classical] | None = None,
        interaction: Sequence[Interaction] | None = None,
        opts: dict[str, Any] | None = None,
        timer: Timers | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Instantiation of the Calculator object.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system
        par : Param
            Full xtb parametrization. Decides energy contributions.
        classical : Sequence[Classical] | None, optional
            Additional classical contributions. Defaults to `None`.
        interaction : Sequence[Interaction] | None, optional
            Additional self-consistent contributions (interactions).
            Defaults to `None`.
        opts : dict[str, Any] | None, optional
            Calculator options. If `None` (default) is given, default options
            are used automatically.
        timer : Timers | None
            Pass an existing `Timers` instance. Defaults to `None`, which
            creates a new timer instance.
        device : torch.device | None, optional
            Device to store the tensor on. If `None` (default), the default
            device is used.
        dtype : torch.dtype | None, optional
            Data type of the tensor. If `None` (default), the data type is
            inferred.
        """
        super().__init__(device, dtype)
        dd = {"device": self.device, "dtype": self.dtype}

        self.timer = Timers("calculator") if timer is None else timer
        self.timer.start("setup calculator")

        # setup calculator options
        opts = opts if opts is not None else {}

        self.batched = numbers.ndim > 1

        self.opts = {
            "fwd_options": {
                "damp": opts.get("damp", defaults.DAMP),
                "maxiter": opts.get("maxiter", defaults.MAXITER),
                "verbose": opts.get("verbose", defaults.XITORCH_VERBOSITY),
            },
            "scf_options": {
                "etemp": opts.get("etemp", defaults.ETEMP),
                "fermi_maxiter": opts.get(
                    "fermi_maxiter",
                    defaults.FERMI_MAXITER,
                ),
                "fermi_thresh": opts.get(
                    "fermi_thresh",
                    defaults.THRESH,
                ),
                "fermi_fenergy_partition": opts.get(
                    "fermi_fenergy_partition",
                    defaults.FERMI_FENERGY_PARTITION,
                ),
                "scf_mode": opts.get("scf_mode", defaults.SCF_MODE),
                "scp_mode": opts.get("scp_mode", defaults.SCP_MODE),
                "mixer": opts.get("mixer", defaults.MIXER),
                "verbosity": opts.get("verbosity", defaults.VERBOSITY),
            },
            "exclude": opts.get("exclude", defaults.EXCLUDE),
            "guess": opts.get("guess", defaults.GUESS),
            "spin": opts.get("spin", defaults.SPIN),
            "int_driver": opts.get("int_driver", defaults.INTDRIVER),
        }

        # set tolerances separately to catch unreasonably small values
        self.set_tol("f_tol", opts.get("xitorch_fatol", defaults.XITORCH_FATOL))
        self.set_tol("x_tol", opts.get("xitorch_xatol", defaults.XITORCH_XATOL))

        self.ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

        # setup self-consistent contributions
        es2 = (
            new_es2(numbers, par, **dd)
            if not any(x in ["all", "es2"] for x in self.opts["exclude"])
            else None
        )
        es3 = (
            new_es3(numbers, par, **dd)
            if not any(x in ["all", "es3"] for x in self.opts["exclude"])
            else None
        )

        if interaction is None:
            self.interactions = InteractionList(es2, es3)
        else:
            self.interactions = InteractionList(es2, es3, *interaction)

        # setup non-self-consistent contributions
        halogen = (
            new_halogen(numbers, par, **dd)
            if not any(x in ["all", "hal"] for x in self.opts["exclude"])
            else None
        )
        dispersion = (
            new_dispersion(numbers, par, **dd)
            if not any(x in ["all", "disp"] for x in self.opts["exclude"])
            else None
        )
        repulsion = (
            new_repulsion(numbers, par, **dd)
            if not any(x in ["all", "rep"] for x in self.opts["exclude"])
            else None
        )

        if classical is None:
            self.classicals = ClassicalList(
                halogen, dispersion, repulsion, timer=self.timer
            )
        else:
            self.classicals = ClassicalList(
                halogen, dispersion, repulsion, *classical, timer=self.timer
            )

        # integral-related setup

        self.hamiltonian = Hamiltonian(numbers, par, self.ihelp, **dd)
        self.integrals = Integrals(**dd)

        # integrals do not work with a batched IndexHelper
        if self.batched:
            self._ihelp = [
                IndexHelper.from_numbers(
                    batch.deflate(number), get_elem_angular(par.element)
                )
                for number in numbers
            ]

        if self.opts["int_driver"] == "libcint":
            self.overlap = OverlapLibcint(numbers, par, self.ihelp, **dd)
            self.basis = Basis(numbers, par, self.ihelp, **dd)
        else:
            self.overlap = Overlap(numbers, par, self.ihelp, **dd)

        # figure out integral level from interactions
        self.set_intlevel(opts.get("int_level", defaults.INTLEVEL))

        self._intdriver = None
        self._positions = None

        self.timer.stop("setup calculator")

    def set_option(self, name: str, value: Any) -> None:
        if name not in self.opts:
            raise KeyError(f"Option '{name}' does not exist.")

        self.opts[name] = value

    def set_tol(self, name: str, value: float) -> None:
        if name not in ("f_tol", "x_tol"):
            raise KeyError(f"Tolerance option '{name}' does not exist.")

        eps = torch.finfo(self.dtype).eps
        if value < eps:
            warnings.warn(
                f"Selected tolerance ({value:.2E}) is smaller than the "
                f"smallest value for the selected dtype ({self.dtype}, "
                f"{eps:.2E}). Switching to {eps:.2E} instead.",
                ToleranceWarning,
            )
            value = eps

        self.opts["fwd_options"][name] = value

    def set_intlevel(self, value: int) -> None:
        if "ElectricField" in self.interactions.labels:
            value = max(2, value)

        self.opts["int_level"] = value

    def driver(
        self, positions: Tensor
    ) -> intor.LibcintWrapper | list[intor.LibcintWrapper]:
        # save current positions to check
        self._positions = positions.detach().clone()

        atombases = self.basis.create_dqc(positions)
        assert isinstance(atombases, list)

        if self.batched:
            return [
                intor.LibcintWrapper(ab, ihelp)
                for ab, ihelp in zip(atombases, self._ihelp)
                if is_basis_list(ab)
            ]

        assert is_basis_list(atombases)
        return intor.LibcintWrapper(atombases, self.ihelp)

    def init_intdriver(self, positions: Tensor):
        if self.opts["int_driver"] != "libcint":
            return

        diff = 0

        # create intor.LibcintWrapper if it does not exist yet
        if self._intdriver is None:
            self._intdriver = self.driver(positions)
        else:
            assert self._positions is not None

            # rebuild driver if positions changed
            diff = (self._positions - positions).abs().sum()
            if diff > 1e-10:
                self._intdriver = self.driver(positions)

        if isinstance(self.overlap, OverlapLibcint):
            if self.overlap.driver is None or diff > 1e-10:
                self.overlap.driver = self._intdriver

    @use_intdriver()
    def singlepoint(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor,
        grad: bool = False,
    ) -> Result:
        """
        Entry point for performing single point calculations.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers.
        positions : Tensor
            Atomic positions.
        chrg : Tensor
            Total charge.
        grad : bool
            Flag for computing nuclear gradient w.r.t. the energy.

        Returns
        -------
        Result
            Results container.
        """
        result = Result(positions, device=self.device, dtype=self.dtype)

        # SELF-CONSISTENT FIELD PROCEDURE
        if not any(x in ["all", "scf"] for x in self.opts["exclude"]):
            # Overlap
            self.timer.start("Overlap")
            overlap = self.overlap.build(positions)
            self.integrals.overlap = overlap
            self.timer.stop("Overlap")

            # dipole integral
            if self.opts["int_driver"] == "libcint" and self.opts["int_level"] > 1:
                self.timer.start("Dipole Integral")
                self.dipole_integral()
                self.timer.stop("Dipole Integral")

            # Hamiltonian
            self.timer.start("h0", "Core Hamiltonian")
            rcov = cov_rad_d3[numbers].to(self.device)
            cn = ncoord.get_coordination_number(
                numbers, positions, ncoord.exp_count, rcov
            )
            hcore = self.hamiltonian.build(positions, overlap, cn)
            self.integrals.hcore = hcore
            self.timer.stop("h0")

            # SCF
            self.timer.start("SCF")

            # Obtain the reference occupations and total number of electrons
            n0 = self.hamiltonian.get_occupation()
            nel = torch.sum(n0, -1) - torch.sum(chrg, -1)

            # get alpha and beta electrons and occupation
            nab = filling.get_alpha_beta_occupation(nel, self.opts["spin"])
            occupation = filling.get_aufbau_occupation(
                hcore.new_tensor(hcore.shape[-1], dtype=torch.int64), nab
            )

            # get caches of all interactions
            icaches = self.interactions.get_cache(
                numbers=numbers, positions=positions, ihelp=self.ihelp
            )
            scf_results = scf.solve(
                numbers,
                positions,
                chrg,
                self.interactions,
                icaches,
                self.ihelp,
                self.opts["guess"],
                self.integrals,
                occupation,
                n0,
                fwd_options=self.opts["fwd_options"],
                scf_options=self.opts["scf_options"],
            )
            self.timer.stop("SCF")

            result.charges = scf_results["charges"]
            result.coefficients = scf_results["coefficients"]
            result.density = scf_results["density"]
            result.emo = scf_results["emo"]
            result.fenergy = scf_results["fenergy"]
            result.hamiltonian = scf_results["hamiltonian"]
            result.occupation = scf_results["occupation"]
            result.potential = scf_results["potential"]
            result.scf += scf_results["energy"]
            result.total += scf_results["energy"] + scf_results["fenergy"]

            if grad is True:
                if len(self.interactions.interactions) > 0:
                    self.timer.start("igrad", "Interaction Gradient")

                    # charges should be detached
                    result.interaction_grad = self.interactions.get_gradient(
                        result.charges, positions, icaches, self.ihelp
                    )
                    result.total_grad += result.interaction_grad
                    self.timer.stop("igrad")
                    # print("grad interaction done")
                    # print(result.interaction_grad)

                self.timer.start("ograd", "Overlap Gradient")
                result.overlap_grad = self.overlap.get_gradient(positions)
                self.timer.stop("ograd")

                self.timer.start("hgrad", "Hamiltonian Gradient")
                wmat = scf.get_density(
                    result.coefficients,
                    result.occupation.sum(-2),
                    emo=result.emo,
                )
                dedcn, dedr = self.hamiltonian.get_gradient(
                    positions,
                    overlap,
                    result.overlap_grad,
                    result.density,
                    wmat,
                    result.potential,
                    cn,
                )

                # CN gradient
                dcndr = ncoord.get_coordination_number_gradient(
                    numbers, positions, ncoord.dexp_count
                )
                dcn = ncoord.get_dcn(dcndr, dedcn)

                # sum up hamiltonian gradient and CN gradient
                result.hamiltonian_grad += dedr + dcn
                result.total_grad += result.hamiltonian_grad
                self.timer.stop("hgrad")

        # CLASSICAL CONTRIBUTIONS
        if len(self.classicals.classicals) > 0:
            ccaches = self.classicals.get_cache(numbers, self.ihelp)
            cenergies = self.classicals.get_energy(positions, ccaches)
            result.cenergies = cenergies
            result.total += torch.stack(list(cenergies.values())).sum(0)

            if grad is True:
                cgradients = self.classicals.get_gradient(cenergies, positions)
                result.cgradients = cgradients
                result.total_grad += torch.stack(list(cgradients.values())).sum(0)

        # TIMERS AND PRINTOUT
        result.integrals = self.integrals
        result.timer = self.timer

        if self.opts["scf_options"]["verbosity"] > 0:
            result.print_energies()
        if self.opts["scf_options"]["verbosity"] > 1:
            print("")
            self.timer.print_times()

        return result

    def dipole_integral(self) -> None:
        # statisfy type checking...
        assert isinstance(self.overlap, OverlapLibcint)
        assert self._intdriver is not None

        def dpint(driver: intor.LibcintWrapper, norm: Tensor) -> Tensor:
            """
            Calculation of dipole integral. The integral is properly
            normalized, using the diagonal of the overlap integral.

            Parameters
            ----------
            driver : intor.LibcintWrapper
                Integral driver (libcint interface).
            norm : Tensor
                Norm of the overlap integral.

            Returns
            -------
            Tensor
                Normalized dipole integral.
            """
            return torch.einsum("xij,i,j->xij", intor.int1e("j", driver), norm, norm)

        if self.batched:
            dpint_list = []

            assert isinstance(self._intdriver, list)
            for _batch, driver in enumerate(self._intdriver):
                d = dpint(driver, batch.deflate(self.overlap.norm[_batch]))
                dpint_list.append(d)

            self.integrals.dipole = batch.pack(dpint_list)
        else:
            assert isinstance(self._intdriver, intor.LibcintWrapper)
            self.integrals.dipole = dpint(self._intdriver, self.overlap.norm)

    def hessian(
        self, numbers: Tensor, positions: Tensor, chrg: Tensor, shape: str = "matrix"
    ) -> Tensor:
        pos = positions.detach().clone()

        def _gradfcn(pos: Tensor) -> Tensor:
            pos.requires_grad_(True)
            result = self.singlepoint(numbers, pos, chrg, grad=True)
            pos.detach_()
            return result.total_grad.detach()

        hess = torch.zeros(
            *(*positions.shape, *positions.shape),
            **{"device": positions.device, "dtype": positions.dtype},
        )

        step = 1.0e-4
        for i in range(numbers.shape[0]):
            for j in range(3):
                pos[i, j] += step
                gr = _gradfcn(pos)

                pos[i, j] -= 2 * step
                gl = _gradfcn(pos)

                pos[i, j] += step
                hess[:, :, i, j] = 0.5 * (gr - gl) / step

        if shape == "matrix":
            hess = hess.reshape(2 * (3 * numbers.shape[-1],))

        return hess

    def vibration(
        self, numbers: Tensor, positions: Tensor, chrg: Tensor
    ) -> tuple[Tensor, Tensor]:
        from ..constants import get_atomic_masses
        from ..utils import eighb

        eps = torch.tensor(
            torch.finfo(positions.dtype).eps,
            device=positions.device,
            dtype=positions.dtype,
        )

        hess = self.hessian(numbers, positions, chrg, shape="matrix")
        masses = get_atomic_masses(numbers, atomic_units=True)
        mass_mat = torch.diag_embed(masses.repeat_interleave(3))

        # instead of calculating and diagonalizing the mass-weighted Hessian,
        # we solve the equivalent general eigenvalue problem Hv=Mve
        evals, evecs = eighb(a=hess, b=mass_mat)

        # vibrational frequencies ω = √λ / (2π) with some additional logic for
        # handling possible negative eigenvalues
        e = torch.sqrt(torch.clamp(evals, min=eps))
        freqs = e * torch.sign(evals) / (2 * torch.pi)

        # reverse the sorting to make it sorted from largest to smallest
        # freqs = torch.flip(freqs, dims=(-1,))
        # evecs = torch.flip(evecs, dims=(-1,))

        return freqs, evecs

    def dipole(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor,
    ) -> Tensor:
        # check if electric field is given in interactions
        if efield.LABEL_EFIELD not in self.interactions.labels:
            raise RuntimeError(
                "Dipole moment requires an electric field. Add the "
                f"'{efield.LABEL_EFIELD}' interaction to the Calculator."
            )

        # retrieve the efield interaction and the field
        ef = self.interactions.get_interaction(efield.LABEL_EFIELD)
        field = ef.field

        if field.requires_grad is False:
            raise RuntimeError("Field tensor needs `requires_grad=True`.")

        result = self.singlepoint(numbers, positions, chrg)
        energy = result.total.sum(-1)

        # Analytical formula:
        #
        # Electric dipole contribution from nuclei: sum_i(r_ik * q_i)
        # qat = self.ihelp.reduce_orbital_to_atom(result.charges.mono)
        # n_dipole = torch.einsum("...ij,...i->...j", positions, qat)
        #
        # Electric dipole contribution from electrons:
        # e_dipole = -torch.einsum("xij,ij->x", self.integrals.dipole, result.density)

        # calculate electric dipole contribution from xtb energy: -de/dE
        dip = -_jac(energy, field)
        return dip

    def quadrupole(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor,
    ) -> Tensor:
        if positions.requires_grad is False:
            raise RuntimeError("Position tensor needs `requires_grad=True`.")

        # check if electric field is given in interactions and retrieve it
        efield = ...
        efield_grad = ...

        energy = self.singlepoint(numbers, positions, chrg).total

        # electric quadrupole contribution from xtb energy: -de/d(dE/dr)
        e_quad = -2 * _jac(energy, efield_grad)
        e_quad = e_quad.view(3, 3)

        # electric quadrupole contribution form nuclei: sum_i(r_ik * Z_i)
        n_quad = torch.einsum(
            "...ij,...ik,...i->...jk",
            positions,
            positions,
            numbers.type(positions.dtype),
        )

        return e_quad + n_quad

    def ir_spectrum(
        self, numbers: Tensor, positions: Tensor, chrg: Tensor
    ) -> tuple[Tensor, Tensor]:
        if positions.requires_grad is False:
            raise RuntimeError("Position tensor needs `requires_grad=True`.")

        # calculate dipole moment first because as it includes a check if the
        # required electric field is given
        mu = self.dipole(numbers, positions, chrg)

        # calculate vibrational frequencies and normal modes and remove
        freqs, modes = self.vibration(numbers, positions, chrg)

        from ..constants import units
        from ..utils.geometry import is_linear_molecule

        is_linear = is_linear_molecule(positions)
        print("\nfreq pre conv", freqs)
        print("units.AU2RCM", units.AU2RCM)

        freqs, modes = _project_freqs(freqs, modes, is_linear=is_linear)
        print("\nfreqs conv 1", units.AU2RCM * freqs)
        freqs = freqs * 1e-2 / units.CODATA.c / 2.4188843265857e-17
        print("freq post conv", freqs)

        # derivative of dipole moment w.r.t. positions
        dmu_dr = _jac(mu, positions)  # (ndim, nat * ndim)
        dmu_dq = torch.matmul(dmu_dr, modes)  # (ndim, nfreqs)
        ir_ints = torch.einsum("...df,...df->...f", dmu_dq, dmu_dq)  # (nfreqs,)

        print("\nir_ints", ir_ints)
        print(units.METER2AU, units.AA2AU)
        print(ir_ints * units.AU2KMMOL)

        return freqs, ir_ints

    def raman_spectrum(
        self, numbers: Tensor, positions: Tensor, chrg: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the frequency and static intensities of Raman spectra.
        Formula taken from `here <https://doi.org/10.1080/00268970701516412>`__.
        Parameters
        ----------
        numbers : Tensor
            Atomic numbers.
        positions : Tensor
            Atomic positions of shape (n, 3).
        chrg : Tensor
            Total charge.
        Returns
        -------
        tuple[Tensor, Tensor]
            Raman frequencies and intensities.
        Raises
        ------
        RuntimeError
            `positions` tensor does not have `requires_grad=True`.
        """
        if positions.requires_grad is False:
            raise RuntimeError("Position tensor needs `requires_grad=True`.")

        # check if electric field is given in interactions and retrieve it
        efield = ...

        # calculate dipole moment first because as it includes a check if the
        # required electric field is given
        mu = self.dipole(numbers, positions, chrg)

        # polarizability: derivative of dipole moment w.r.t. efield and positions
        dmu_de = _jac(mu, efield)  # (ndim, ndim)
        alpha = _jac(dmu_de, positions)  # (ndim, ndim, natoms * ndim)

        # get the vibrational frequencies and normal modes
        # freqs: (nmodes,)
        # normal_modes: (natoms * ndim, nmodes)
        freqs, normal_modes = _only_positive_freqs(
            *self.vibration(numbers, positions, chrg)
        )

        alphaq = torch.matmul(alpha, normal_modes)  # (ndim, ndim, nmodes)

        # Eq.3 with alpha' = a
        a = torch.einsum("...iij->...j")

        # Eq.4 with (gamma')^2 = g = 0.5 * (g1 + g2 + g3 + g4)
        g1 = (alphaq[0, 0] - alphaq[1, 1]) ** 2
        g2 = (alphaq[0, 0] - alphaq[2, 2]) ** 2
        g3 = (alphaq[2, 2] - alphaq[1, 1]) ** 2
        g4 = alphaq[0, 1] ** 2 + alphaq[1, 2] ** 2 + alphaq[2, 0] ** 2
        g = 0.5 * (g1 + g2 + g3 + g4)

        # Eq.1 (the 1/3 from Eq.3 is squared and reduces the 45)
        raman_ints = 5 * torch.pow(a, 2.0) + 7 * g
        return freqs, raman_ints


def _project_freqs(
    freqs: Tensor, modes: Tensor, is_linear: bool = False
) -> tuple[Tensor, Tensor]:
    skip = 5 if is_linear is True else 6
    freqs = freqs[skip:]  # (nfreqs,)
    modes = modes[:, skip:]  # (natoms * ndim, nfreqs)
    return freqs, modes


def _jac(
    a: Tensor,
    b: Tensor,
    create_graph: bool | None = None,
    retain_graph: bool = True,
) -> Tensor:
    # catch missing gradients (e.g., halogen bond correction evaluates to
    # zero if no donors/acceptors are present)
    if a.grad_fn is None:
        return torch.zeros(
            (*a.shape, b.numel()),
            dtype=b.dtype,
            device=b.device,
        )

    if create_graph is None:
        create_graph = torch.is_grad_enabled()
    assert create_graph is not None

    aflat = a.reshape(-1)
    anumel, bnumel = a.numel(), b.numel()
    res = torch.empty(
        (anumel, bnumel),
        dtype=a.dtype,
        device=a.device,
    )

    for i in range(aflat.numel()):
        (g,) = torch.autograd.grad(
            aflat[i],
            b,
            create_graph=create_graph,
            retain_graph=retain_graph,
        )
        res[i] = g.reshape(-1)

    return res.reshape((*a.shape, bnumel))
