"""
Base calculator for the extended tight-binding model.
"""

from __future__ import annotations

import logging

import torch
from tad_mctc.convert import any_to_tensor

from .. import integral as ints
from .. import ncoord, properties, scf
from .._types import Any, Callable, Literal, Sequence, Tensor, TensorLike
from ..basis import IndexHelper
from ..classical import (
    Classical,
    ClassicalList,
    Halogen,
    Repulsion,
    new_halogen,
    new_repulsion,
)
from ..config import Config
from ..constants import defaults
from ..coulomb import new_es2, new_es3
from ..data import cov_rad_d3
from ..dispersion import Dispersion, new_dispersion
from ..exceptions import DtypeError
from ..interaction import Charges, Interaction, InteractionList, Potential
from ..interaction.external import field as efield
from ..interaction.external import fieldgrad as efield_grad
from ..io import OutputHandler
from ..param import Param, get_elem_angular
from ..timing import timer
from ..utils import _jac
from ..wavefunction import filling

__all__ = ["Calculator", "Result"]


logger = logging.getLogger(__name__)


# class CalculatorFunction(Protocol):
#     def __call__(
#         self: "Calculator",
#         numbers: Tensor,
#         positions: Tensor,
#         chrg: Tensor | float | int = defaults.CHRG,
#         spin: Tensor | float | int | None = defaults.SPIN,
#         **kwargs: Any
#     ) -> tuple[torch.Tensor, Tensor]:
#         ...


def requires_positions_grad(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    def wrapper(
        self: Calculator,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        if not positions.requires_grad:
            raise RuntimeError(
                f"Position tensor needs `requires_grad=True` in '{func.__name__}'."
            )

        return func(self, numbers, positions, chrg, spin, **kwargs)

    return wrapper


def requires_efield(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    def wrapper(
        self: Calculator,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        if efield.LABEL_EFIELD not in self.interactions.labels:
            raise RuntimeError(
                f"{func.__name__} requires an electric field. Add the "
                f"'{efield.LABEL_EFIELD}' interaction to the Calculator."
            )
        return func(self, numbers, positions, chrg, spin, **kwargs)

    return wrapper


def requires_efield_grad(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    def wrapper(
        self: Calculator,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        ef = self.interactions.get_interaction(efield.LABEL_EFIELD)
        if not ef.field.requires_grad:
            raise RuntimeError(
                f"Field tensor needs `requires_grad=True` in '{func.__name__}'."
            )
        return func(self, numbers, positions, chrg, spin, **kwargs)

    return wrapper


def requires_efg(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    def wrapper(
        self: Calculator,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        if efield_grad.LABEL_EFIELD_GRAD not in self.interactions.labels:
            raise RuntimeError(
                f"{func.__name__} requires an electric field. Add the "
                f"'{efield_grad.LABEL_EFIELD_GRAD}' interaction to the "
                "Calculator."
            )
        return func(self, numbers, positions, chrg, spin, **kwargs)

    return wrapper


def requires_efg_grad(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    def wrapper(
        self: Calculator,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        efg = self.interactions.get_interaction(efield_grad.LABEL_EFIELD_GRAD)
        if not efg.field_grad.requires_grad:
            raise RuntimeError("Field gradient tensor needs `requires_grad=True`.")
        return func(self, numbers, positions, chrg, spin, **kwargs)

    return wrapper


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

    integrals: ints.Integrals
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

    opts: Config
    """Calculator configuration."""

    class Cache(TensorLike):
        """
        Cache for Calculator that extends TensorLike.

        This class provides caching functionality for storing multiple calculation results.
        """

        __slots__ = ["energy", "forces"]

        def __init__(
            self,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            """
            Initialize the Cache class with optional device and dtype settings.

            Parameters
            ----------
            device : torch.device, optional
                The device on which the tensors are stored.
            dtype : torch.dtype, optional
                The data type of the tensors.
            """
            super().__init__(device=device, dtype=dtype)
            self.energy = None
            self.forces = None

        def __getitem__(self, key: str) -> Tensor:
            """
            Get an item from the cache.

            Parameters
            ----------
            key : str
                The key of the item to retrieve.

            Returns
            -------
            Tensor or None
                The value associated with the key, if it exists.
            """
            if key in self.__slots__:
                return getattr(self, key)
            raise KeyError(f"Key '{key}' not found in Cache.")

        def __setitem__(self, key: str, value: Tensor) -> None:
            """
            Set an item in the cache.

            Parameters
            ----------
            key : str
                The key of the item to set.
            value : Tensor
                The value to be associated with the key.
            """
            if key in self.__slots__:
                setattr(self, key, value)
            else:
                raise KeyError(f"Key '{key}' cannot be set in Cache.")

        def __contains__(self, key: str) -> bool:
            """
            Check if a key is in the cache.

            Parameters
            ----------
            key : str
                The key to check in the cache.

            Returns
            -------
            bool
                True if the key is in the cache, False otherwise
            """
            return key in self.__slots__ and getattr(self, key) is not None

        def clear(self, key: str | None = None) -> None:
            """
            Clear the cached values.
            """
            if key is not None:
                setattr(self, key, None)
            else:
                for key in self.__slots__:
                    setattr(self, key, None)

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        *,
        classical: Sequence[Classical] = [],
        interaction: Sequence[Interaction] = [],
        opts: dict[str, Any] | Config | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Instantiation of the Calculator object.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        par : Param
            Representation of an extended tight-binding model (full xtb
            parametrization). Decides energy contributions.
        classical : Sequence[Classical] | None, optional
            Additional classical contributions. Defaults to `None`.
        interaction : Sequence[Interaction] | None, optional
            Additional self-consistent contributions (interactions).
            Defaults to `None`.
        opts : dict[str, Any] | None, optional
            Calculator options. If `None` (default) is given, default options
            are used automatically.
        device : torch.device | None, optional
            Device to store the tensor on. If `None` (default), the default
            device is used.
        dtype : torch.dtype | None, optional
            Data type of the tensor. If `None` (default), the data type is
            inferred.
        """
        timer.start("setup calculator")

        allowed_dtypes = (torch.long, torch.int16, torch.int32, torch.int64)
        if numbers.dtype not in allowed_dtypes:
            raise DtypeError(
                "Dtype of atomic numbers must be one of the following to allow "
                f"indexing: '{', '.join([str(x) for x in allowed_dtypes])}', "
                f"but is '{numbers.dtype}'"
            )

        super().__init__(device, dtype)
        dd = {"device": self.device, "dtype": self.dtype}
        self.cache = self.Cache(**dd)

        # setup calculator options
        opts = opts if opts is not None else {}

        if isinstance(opts, dict):
            OutputHandler.verbosity = opts.pop("verbosity", None)
            opts = Config(**opts, **dd)

        self.opts = opts

        self.batched = numbers.ndim > 1

        self.ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

        # setup self-consistent contributions
        es2 = (
            new_es2(numbers, par, **dd)
            if not any(x in ["all", "es2"] for x in self.opts.exclude)
            else None
        )
        es3 = (
            new_es3(numbers, par, **dd)
            if not any(x in ["all", "es3"] for x in self.opts.exclude)
            else None
        )

        self.interactions = InteractionList(es2, es3, *interaction)

        # setup non-self-consistent contributions
        halogen = (
            new_halogen(numbers, par, **dd)
            if not any(x in ["all", "hal"] for x in self.opts.exclude)
            else None
        )
        dispersion = (
            new_dispersion(numbers, par, **dd)
            if not any(x in ["all", "disp"] for x in self.opts.exclude)
            else None
        )
        repulsion = (
            new_repulsion(numbers, par, **dd)
            if not any(x in ["all", "rep"] for x in self.opts.exclude)
            else None
        )

        self.classicals = ClassicalList(halogen, dispersion, repulsion, *classical)

        #############
        # INTEGRALS #
        #############

        # figure out integral level from interactions
        self.intlevel = defaults.INTLEVEL
        if efield.LABEL_EFIELD in self.interactions.labels:
            self.intlevel = max(ints.INTLEVEL_DIPOLE, self.intlevel)
        if efield_grad.LABEL_EFIELD_GRAD in self.interactions.labels:
            self.intlevel = max(ints.INTLEVEL_QUADRUPOLE, self.intlevel)

        # setup integral
        driver = self.opts.ints.driver
        self.integrals = ints.Integrals(numbers, par, self.ihelp, driver=driver, **dd)

        if self.intlevel >= ints.INTLEVEL_OVERLAP:
            self.integrals.hcore = ints.Hamiltonian(numbers, par, self.ihelp, **dd)
            self.integrals.overlap = ints.Overlap(driver=driver, **dd)

        if self.intlevel >= ints.INTLEVEL_DIPOLE:
            self.integrals.dipole = ints.Dipole(driver=driver, **dd)

        if self.intlevel >= ints.INTLEVEL_QUADRUPOLE:
            self.integrals.quadrupole = ints.Quadrupole(driver=driver, **dd)

        timer.stop("setup calculator")

    def reset(self) -> None:
        self.interactions.reset_all()
        self.classicals.reset_all()

    def singlepoint(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        grad: bool = False,
    ) -> Result:
        """
        Entry point for performing single point calculations.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to 0.
        grad : bool, optional
            Flag for computing nuclear gradient w.r.t. the energy.

        Returns
        -------
        Result
            Results container.
        """
        chrg = any_to_tensor(chrg, **self.dd)
        if spin is not None:
            spin = any_to_tensor(spin, **self.dd)

        result = Result(positions, device=self.device, dtype=self.dtype)

        # CLASSICAL CONTRIBUTIONS
        if len(self.classicals.components) > 0:
            ccaches = self.classicals.get_cache(numbers, self.ihelp)
            cenergies = self.classicals.get_energy(positions, ccaches)
            result.cenergies = cenergies
            result.total += torch.stack(list(cenergies.values())).sum(0)

            if grad is True:
                cgradients = self.classicals.get_gradient(cenergies, positions)
                result.cgradients = cgradients
                result.total_grad += torch.stack(list(cgradients.values())).sum(0)

        # SELF-CONSISTENT FIELD PROCEDURE
        if not any(x in ["all", "scf"] for x in self.opts.exclude):
            # overlap integral
            timer.start("Overlap")
            self.integrals.build_overlap(positions)
            timer.stop("Overlap")

            # dipole integral
            if self.intlevel >= ints.INTLEVEL_DIPOLE:
                timer.start("Dipole Integral")
                self.integrals.build_dipole(positions)
                timer.stop("Dipole Integral")

            # quadrupole integral
            if self.intlevel >= ints.INTLEVEL_QUADRUPOLE:
                timer.start("Quadrupole Integral")
                self.integrals.build_quadrupole(positions)
                timer.stop("Quadrupole Integral")

            # TODO: Think about handling this case
            if self.integrals.hcore is None:
                raise RuntimeError
            if self.integrals.overlap is None:
                raise RuntimeError

            # Core Hamiltonian integral (requires overlap internally!)
            timer.start("h0", "Core Hamiltonian")
            rcov = cov_rad_d3.to(self.device)[numbers]
            cn = ncoord.get_coordination_number(
                numbers, positions, ncoord.exp_count, rcov
            )
            hcore = self.integrals.build_hcore(positions, cn=cn)
            timer.stop("h0")

            # SCF
            timer.start("SCF")

            # Obtain the reference occupations and total number of electrons
            n0 = self.integrals.hcore.integral.get_occupation()
            nel = torch.sum(n0, -1) - torch.sum(chrg, -1)

            # get alpha and beta electrons and occupation
            nab = filling.get_alpha_beta_occupation(nel, spin)
            occupation = filling.get_aufbau_occupation(
                torch.tensor(hcore.shape[-1], device=self.device, dtype=torch.int64),
                nab,
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
                self.opts.scf,
                self.integrals.matrices,
                occupation,
                n0,
            )
            timer.stop("SCF")

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
                if len(self.interactions.components) > 0:
                    timer.start("igrad", "Interaction Gradient")

                    # charges should be detached
                    result.interaction_grad = self.interactions.get_gradient(
                        result.charges, positions, icaches, self.ihelp
                    )
                    result.total_grad += result.interaction_grad
                    timer.stop("igrad")

                timer.start("ograd", "Overlap Gradient")
                result.overlap_grad = self.integrals.grad_overlap(positions)
                timer.stop("ograd")

                timer.start("hgrad", "Hamiltonian Gradient")
                wmat = scf.get_density(
                    result.coefficients,
                    result.occupation.sum(-2),
                    emo=result.emo,
                )
                dedcn, dedr = self.integrals.hcore.integral.get_gradient(
                    positions,
                    self.integrals.matrices.overlap,
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
                timer.stop("hgrad")

        # TIMERS AND PRINTOUT
        result.integrals = self.integrals

        return result

    def energy(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_cache: bool = False,
    ) -> Tensor:
        if use_cache:
            if "energy" in self.cache:
                return self.cache["energy"].sum(-1)

        result = self.singlepoint(numbers, positions, chrg, spin).total
        self.cache["energy"] = result
        return result.sum(-1)

    @requires_positions_grad
    def forces(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int | None = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
    ) -> Tensor:
        logger.debug(f"Autodiff Forces: Starting Calculation.")

        # pylint: disable=import-outside-toplevel
        from tad_mctc.autograd import jacrev

        # jacrev requires a scalar from `self.energy`!
        jac_func = jacrev(self.energy, argnums=1)
        jac = jac_func(numbers, positions, chrg, spin)
        assert isinstance(jac, Tensor)

        logger.debug("Autodiff Forces: All finished.")
        return -jac

    def forces_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = 1.0e-5,
    ) -> Tensor:
        # important: detach for gradient
        pos = positions.detach().clone()

        # turn off printing in numerical calc
        tmp = OutputHandler.verbosity
        OutputHandler.verbosity = 0

        jac = torch.zeros(pos.shape, **self.dd)

        logger.debug(f"Numerical Forces: Starting build ({jac.shape})")

        count = 1
        nsteps = 3 * numbers.shape[-1]
        for i in range(numbers.shape[-1]):
            for j in range(3):
                pos[..., i, j] += step_size
                gr = self.energy(numbers, pos, chrg, spin)

                pos[..., i, j] -= 2 * step_size
                gl = self.energy(numbers, pos, chrg, spin)

                pos[..., i, j] += step_size
                jac[..., i, j] = 0.5 * (gr - gl) / step_size

                logger.debug(f"Numerical Forces: step {count}/{nsteps}")
                count += 1

        logger.debug("Numerical Forces: All finished.")

        OutputHandler.verbosity = tmp
        return -jac

    @requires_positions_grad
    def hessian(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        matrix: bool = False,
    ) -> Tensor:
        """
        Calculation of the nuclear (autodiff) Hessian with functorch.

        Note
        ----
        The `jacrev` function of `functorch` requires scalars for the expected
        behavior, i.e., the nuclear Hessian only acquires the expected shape of
        `(nat, 3, nat, 3)` if the energy is provided as a scalar value.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        matrix : bool, optional
            Whether to reshape the Hessian to a matrix, i.e., (nat*3, nat*3).
            Defaults to `False`.

        Returns
        -------
        Tensor
            Hessian matrix.

        Raises
        ------
        RuntimeError
            Positions tensor does not have `requires_grad=True`.
        """
        logger.debug(f"Autodiff Hessian: Starting Calculation.")

        # pylint: disable=import-outside-toplevel
        from tad_mctc.autograd import jacrev

        # jacrev requires a scalar from `self.energy`!
        hess_func = jacrev(jacrev(self.energy, argnums=1), argnums=1)
        hess = hess_func(numbers, positions, chrg, spin)
        assert isinstance(hess, Tensor)

        # reshape (nb, nat, 3, nat, 3) to (nb, nat*3, nat*3)
        if matrix is True:
            s = [*numbers.shape[:-1], *2 * [3 * numbers.shape[-1]]]
            hess = hess.reshape(*s)

        logger.debug("Autodiff Hessian: All finished.")
        return hess

    def hessian2(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        matrix: bool = False,
    ) -> Tensor:
        if positions.requires_grad is False:
            raise RuntimeError("Position tensor needs `requires_grad=True`.")

        logger.debug(f"Autodiff Hessian: Starting Calculation.")

        # pylint: disable=import-outside-toplevel
        from tad_mctc.autograd import hessian

        # jacrev requires a scalar from `self.energy`!
        hess = hessian(self.energy, (numbers, positions, chrg, spin), argnums=1)
        assert isinstance(hess, Tensor)

        # reshape (nb, nat, 3, nat, 3) to (nb, nat*3, nat*3)
        if matrix is True:
            s = [*numbers.shape[:-1], *2 * [3 * numbers.shape[-1]]]
            hess = hess.reshape(*s)

        logger.debug("Autodiff Hessian: All finished.")
        return hess

    def hessian_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = 1.0e-5,
        matrix: bool = False,
    ) -> Tensor:
        """
        Numerical Hessian.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        chrg : Tensor | float | int | str | None
            Total charge. Defaults to `None`.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        matrix : bool, optional
            Whether to reshape the Hessian to a matrix, i.e., (nat*3, nat*3).
            Defaults to `False`.

        Returns
        -------
        Tensor
            Hessian.
        """
        # pylint: disable=import-outside-toplevel
        import gc

        def _gradfcn(pos: Tensor) -> Tensor:
            pos.requires_grad_(True)
            result = self.singlepoint(numbers, pos, chrg, spin, grad=True)
            pos.detach_()
            return result.total_grad.detach()

        # imortant: detach for gradient
        pos = positions.detach().clone()

        hess = torch.zeros(
            *(*positions.shape, *positions.shape[-2:]),
            **{"device": positions.device, "dtype": positions.dtype},
        )

        # turn off printing in numerical hessian
        OutputHandler.temporary_disable_on()
        logger.debug(f"Numerical Hessian: Starting build (size {hess.shape})")

        count = 1
        nsteps = 3 * numbers.shape[-1]
        for i in range(numbers.shape[-1]):
            for j in range(3):
                pos[..., i, j] += step_size
                gr = _gradfcn(pos)

                pos[..., i, j] -= 2 * step_size
                gl = _gradfcn(pos)

                pos[..., i, j] += step_size
                hess[..., :, :, i, j] = 0.5 * (gr - gl) / step_size

                logger.debug(f"Numerical Hessian: step {count}/{nsteps}")
                count += 1

            gc.collect()
        gc.collect()

        # reshape (nb, nat, 3, nat, 3) to (nb, nat*3, nat*3)
        if matrix is True:
            s = [*numbers.shape[:-1], *2 * [3 * numbers.shape[-1]]]
            hess = hess.reshape(*s)

        logger.debug("Numerical Hessian: All finished.")
        OutputHandler.temporary_disable_off()

        return hess

    def vibration(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        project_translational: bool = True,
        project_rotational: bool = True,
    ) -> tuple[Tensor, Tensor]:
        hess = self.hessian(numbers, positions, chrg, spin)
        return properties.frequencies(
            numbers,
            positions,
            hess,
            project_translational=project_translational,
            project_rotational=project_rotational,
        )

    def vibration_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        project_translational: bool = True,
        project_rotational: bool = True,
    ) -> tuple[Tensor, Tensor]:
        hess = self.hessian_numerical(numbers, positions, chrg, spin)
        return properties.frequencies(
            numbers,
            positions,
            hess,
            project_translational=project_translational,
            project_rotational=project_rotational,
        )

    @requires_efield
    @requires_efield_grad
    def dipole(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
    ) -> Tensor:
        field = self.interactions.get_interaction(efield.LABEL_EFIELD).field

        if use_functorch is True:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            def wrapped_energy(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return self.energy(numbers, positions, chrg, spin, use_cache=False)

            dip = jacrev(wrapped_energy)(field)
            assert isinstance(dip, Tensor)
        else:
            # calculate electric dipole contribution from xtb energy: -de/dE
            energy = self.energy(numbers, positions, chrg, spin)
            dip = _jac(energy, field)

        return -dip

    @requires_efield
    def dipole_analytical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
    ) -> Tensor:
        # run single point and check if integral is populated
        result = self.singlepoint(numbers, positions, chrg, spin)
        dipint = self.integrals.dipole
        if dipint is None:
            raise RuntimeError(
                "Dipole moment requires a dipole integral. They should "
                f"be added automatically if the '{efield.LABEL_EFIELD}' "
                "interaction is added to the Calculator."
            )
        if dipint.matrix is None:
            raise RuntimeError(
                "Dipole moment requires a dipole integral. They should "
                f"be added automatically if the '{efield.LABEL_EFIELD}' "
                "interaction is added to the Calculator."
            )

        # dip = properties.dipole(
        # numbers, positions, result.density, self.integrals.dipole
        # )
        qat = self.ihelp.reduce_orbital_to_atom(result.charges.mono)
        dip = properties.moments.dipole(qat, positions, result.density, dipint.matrix)
        return -dip

    @requires_efield
    def dipole_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = 1.0e-5,
    ) -> Tensor:
        # retrieve the efield interaction and the field and detach for gradient
        ef = self.interactions.get_interaction(efield.LABEL_EFIELD)
        field = ef.field.detach().clone()
        self.interactions.update_efield(field=field)

        deriv = torch.zeros(*(*numbers.shape[:-1], 3), **self.dd)

        # turn off printing in numerical hessian
        OutputHandler.temporary_disable_on()
        logger.debug(f"Numerical Dipole: Starting build ({deriv.shape}).")

        count = 1
        for i in range(3):
            field[..., i] += step_size
            self.interactions.update_efield(field=field)
            gr = self.energy(numbers, positions, chrg, spin)

            field[..., i] -= 2 * step_size
            self.interactions.update_efield(field=field)
            gl = self.energy(numbers, positions, chrg, spin)

            field[..., i] += step_size
            self.interactions.update_efield(field=field)
            deriv[..., i] = 0.5 * (gr - gl) / step_size

            logger.debug(f"Numerical Dipole: step {count}/{3}")
            count += 1

        logger.debug("Numerical Dipole: All finished.")
        OutputHandler.temporary_disable_off()

        return -deriv

    @requires_positions_grad
    def dipole_deriv(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
    ) -> Tensor:
        r"""
        Calculate cartesian dipole derivative :math:`\mu'`.

        .. math::

            \mu' = \dfrac{\partial \mu}{\partial R} = \dfrac{\partial^2 E}{\partial F \partial R}

        One can calculate the Jacobian either row-by-row using the standard
        `torch.autograd.grad` with unit vectors in the VJP (see `here`_) or
        using `torch.func`'s function transforms (e.g., `jacrev`).

        .. _here: https://pytorch.org/functorch/stable/notebooks/jacobians_hessians.html#computing-the-jacobian

        Note
        ----
        Using `torch.func`'s function transforms can apparently be only used
        once. Hence, for example, the Hessian and the dipole derivatives cannot
        be both calculated with functorch.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.

        Returns
        -------
        Tensor
            Cartesian dipole derivative of shape `(3, nat, 3)`.
        """
        if use_functorch is True:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            # d(3) / d(nat, 3) = (3, nat, 3)
            dmu_dr = jacrev(self.dipole, argnums=1)(numbers, positions, chrg, spin)
            assert isinstance(dmu_dr, Tensor)
        else:
            mu = self.dipole(numbers, positions, chrg, spin)

            # (3, 3*nat) -> (3, nat, 3)
            dmu_dr = _jac(mu, positions).reshape((3, *positions.shape[-2:]))

        return dmu_dr

    def dipole_deriv_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = 1.0e-5,
    ) -> Tensor:
        # turn off printing in numerical calcs
        OutputHandler.temporary_disable_on()
        logger.debug("Dipole derivative: Start.")

        # pylint: disable=import-outside-toplevel
        import gc

        # important: use new/separate position tensor
        pos = positions.detach().clone()

        dmu_dr = torch.zeros((3, *positions.shape[-2:]), **self.dd)  # (3, n, 3)

        logger.debug(
            "Dipole derivative: Start building numerical dipole derivative "
            f"({dmu_dr.shape})."
        )

        count = 1
        nsteps = 3 * numbers.shape[-1]
        for i in range(numbers.shape[-1]):
            for j in range(3):
                pos[i, j] += step_size
                r = self.dipole(numbers, pos, chrg, spin)

                pos[i, j] -= 2 * step_size
                l = self.dipole(numbers, pos, chrg, spin)

                pos[i, j] += step_size
                dmu_dr[:, i, j] = 0.5 * (r - l) / step_size

                logger.debug("Dipole derivative: Step " f"{count}/{nsteps}")
                count += 1

            gc.collect()
        gc.collect()

        # dmu_dr = dmu_dr.view(3, -1)

        logger.debug("Dipole derivative: All finished.")
        OutputHandler.temporary_disable_off()

        return dmu_dr

    @requires_efg
    @requires_efg_grad
    def quadrupole(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
    ) -> Tensor:
        # retrieve the efield interaction and the field
        efg = self.interactions.get_interaction(efield_grad.LABEL_EFIELD_GRAD)
        field_grad = efg.field_grad

        if use_functorch is True:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            def wrapped_energy(f: Tensor) -> Tensor:
                self.interactions.update_efield_grad(field_grad=f)
                return self.energy(numbers, positions, chrg, spin, use_cache=False)

            e_quad = jacrev(wrapped_energy)(field_grad)
            assert isinstance(e_quad, Tensor)
        else:
            energy = self.energy(numbers, positions, chrg, spin)
            e_quad = _jac(energy, field_grad)

        e_quad = e_quad.view(3, 3)
        print("quad_moment\n", e_quad)
        cart = torch.empty((6), **self.dd)
        cart[..., 0] = e_quad[..., 0, 0]
        cart[..., 1] = e_quad[..., 1, 0]
        cart[..., 2] = e_quad[..., 1, 1]
        cart[..., 3] = e_quad[..., 2, 0]
        cart[..., 4] = e_quad[..., 2, 1]
        cart[..., 5] = e_quad[..., 2, 2]

        return cart

        print("")
        print("quad_moment\n", e_quad)

        e_quad = e_quad.view(3, 3)
        print("")
        print("")

        print("quad_moment", e_quad.shape)

        cart = torch.empty((6), **self.dd)

        tr = 0.5 * torch.einsum("...ii->...", e_quad)
        print("tr", tr)
        cart[..., 0] = 1.5 * e_quad[..., 0, 0] - tr
        cart[..., 1] = 3.0 * e_quad[..., 1, 0]
        cart[..., 2] = 1.5 * e_quad[..., 1, 1] - tr
        cart[..., 3] = 3.0 * e_quad[..., 2, 0]
        cart[..., 4] = 3.0 * e_quad[..., 2, 1]
        cart[..., 5] = 1.5 * e_quad[..., 2, 2] - tr

        print("cart\n", cart)

        # electric quadrupole contribution form nuclei: sum_i(r_ik * Z_i)
        n_quad = torch.einsum(
            "...ij,...ik,...i->...jk",
            positions,
            positions,
            numbers.type(positions.dtype),
        )

        print(n_quad)
        print("\ne_quad + n_quad")
        print(e_quad + n_quad)
        print("\ne_quad")
        print(e_quad)
        print("")

        return cart

    @requires_efg
    def quadrupole_analytical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor,
    ) -> Tensor:
        result = self.singlepoint(numbers, positions, chrg)
        if result.charges.dipole is None:
            raise RuntimeError(
                "Dipole charges were not calculated but are required for "
                "quadrupole moment."
            )
        if result.charges.quad is None:
            raise RuntimeError(
                "Quadrupole charges were not calculated but are required "
                "for quadrupole moment."
            )

        qat = self.ihelp.reduce_orbital_to_atom(result.charges.mono)
        dpat = result.charges.dipole
        qpat = result.charges.quad

        return properties.quadrupole(qat, dpat, qpat, positions)

    @requires_efg
    def quadrupole_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = 1.0e-5,
    ) -> Tensor:
        # retrieve the efg interaction and the field gradient and detach
        efg = self.interactions.get_interaction(efield_grad.LABEL_EFIELD_GRAD)
        _field_grad = efg.field_grad.clone()
        field_grad = efg.field_grad.detach().clone()
        self.interactions.update_efield_grad(field_grad=field_grad)

        deriv = torch.zeros(*(*numbers.shape[:-1], 3, 3), **self.dd)

        # turn off printing in numerical derivatives
        OutputHandler.temporary_disable_on()
        logger.debug(f"Numerical Quadrupole: Starting build ({deriv.shape}).")

        count = 1
        for i in range(3):
            for j in range(3):
                field_grad[..., i, j] += step_size
                self.interactions.update_efield_grad(field_grad=field_grad)
                gr = self.energy(numbers, positions, chrg, spin)

                field_grad[..., i, j] -= 2 * step_size
                self.interactions.update_efield_grad(field_grad=field_grad)
                gl = self.energy(numbers, positions, chrg, spin)

                field_grad[..., i, j] += step_size
                self.interactions.update_efield_grad(field_grad=field_grad)
                deriv[..., i, j] = 0.5 * (gr - gl) / step_size

                logger.debug(f"Numerical Quadrupole: step {count}/{3}")
                count += 1

        # reset
        self.interactions.update_efield_grad(field_grad=_field_grad)

        logger.debug("Numerical Quadrupole: All finished.")
        OutputHandler.temporary_disable_off()

        return deriv

    @requires_efield
    @requires_efield_grad
    def polarizability(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
        derived_quantity: Literal["energy", "dipole"] = "dipole",
    ) -> Tensor:
        r"""
        Calculate the polarizability tensor :math:`\alpha`.

        .. math::

            \alpha = \dfrac{\partial \mu}{\partial F} = \dfrac{\partial^2 E}{\partial^2 F}

        One can calculate the Jacobian either row-by-row using the standard
        `torch.autograd.grad` with unit vectors in the VJP (see `here`_) or
        using `torch.func`'s function transforms (e.g., `jacrev`).

        .. _here: https://pytorch.org/functorch/stable/notebooks/jacobians_hessians.html#computing-the-jacobian

        Note
        ----
        Using `torch.func`'s function transforms can apparently be only used
        once. Hence, for example, the Hessian and the polarizability cannot
        be both calculated with functorch.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.
        derived_quantity: Literal['energy', 'dipole'], optional
            Which derivative to calculate for the polarizability, i.e.,
            derivative of dipole moment or energy w.r.t field.

        Returns
        -------
        Tensor
            Polarizability tensor of shape `(3, 3)`.
        """
        # retrieve the efield interaction and the field
        field = self.interactions.get_interaction(efield.LABEL_EFIELD).field

        if use_functorch is False:
            mu = self.dipole(numbers, positions, chrg, spin)
            return _jac(mu, field)

        # pylint: disable=import-outside-toplevel
        from tad_mctc.autograd import jacrev

        if derived_quantity == "dipole":

            def wrapped_dipole(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return self.dipole(numbers, positions, chrg, spin)

            alpha = jacrev(wrapped_dipole)(field)
            assert isinstance(alpha, Tensor)
        elif derived_quantity == "energy":

            def wrapped_energy(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return self.energy(numbers, positions, chrg, spin)

            alpha = jacrev(jacrev(wrapped_energy))(field)
            assert isinstance(alpha, Tensor)

            #
            alpha = -alpha
        else:
            raise ValueError(
                f"Unknown `derived_quantity` '{derived_quantity}'. The "
                "polarizability can be calculated as the derivative of the "
                "'dipole' moment or the 'energy'."
            )

        # 3x3 polarizability tensor
        return alpha

    @requires_efield
    def polarizability_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = 1.0e-5,
    ) -> Tensor:
        """
        Numerical polarizability.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        chrg : Tensor | float | int | str | None
            Total charge. Defaults to `None`.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        step_size : float | int, optional
            Step size for the numerical derivative.

        Returns
        -------
        Tensor
            Polarizability tensor.
        """
        # retrieve the efield interaction and the field and detach for gradient
        ef = self.interactions.get_interaction(efield.LABEL_EFIELD)
        _field = ef.field.clone()
        field = ef.field.detach().clone()
        self.interactions.update_efield(field=field)

        # pylint: disable=import-outside-toplevel
        import gc

        deriv = torch.zeros(*(*numbers.shape[:-1], 3, 3), **self.dd)

        # turn off printing in numerical derivative
        OutputHandler.temporary_disable_on()
        logger.debug(f"Numerical Polarizability: Starting build ({deriv.shape})")

        count = 1
        for i in range(3):
            field[..., i] += step_size
            self.interactions.update_efield(field=field)
            gr = self.dipole_analytical(numbers, positions, chrg, spin)

            field[..., i] -= 2 * step_size
            self.interactions.update_efield(field=field)
            gl = self.dipole_analytical(numbers, positions, chrg, spin)

            field[..., i] += step_size
            self.interactions.update_efield(field=field)
            deriv[..., :, i] = 0.5 * (gr - gl) / step_size

            logger.debug(f"Numerical Polarizability: step {count}/{3}")
            count += 1

            gc.collect()

        logger.debug("Numerical Polarizability: All finished.")
        OutputHandler.temporary_disable_off()

        # explicitly update field (otherwise gradient is missing)
        self.interactions.reset_efield()
        self.interactions.update_efield(field=_field)

        return -deriv

    @requires_efield
    @requires_positions_grad
    def pol_deriv(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
    ) -> Tensor:
        r"""
        Calculate the cartesian polarizability derivative :math:`\chi`.

        .. math::

            \chi = \alpha' = \dfrac{\partial \alpha}{\partial R} = \dfrac{\partial^2 \mu}{\partial F \partial R} = \dfrac{\partial^3 E}{\partial^2 F \partial R}

        One can calculate the Jacobian either row-by-row using the standard
        `torch.autograd.grad` with unit vectors in the VJP (see `here`_) or
        using `torch.func`'s function transforms (e.g., `jacrev`).

        .. _here: https://pytorch.org/functorch/stable/notebooks/jacobians_hessians.html#computing-the-jacobian

        Note
        ----
        Using `torch.func`'s function transforms can apparently be only used
        once. Hence, for example, the Hessian and the polarizability cannot
        be both calculated with functorch.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.

        Returns
        -------
        Tensor
            Polarizability derivative shape `(3, 3, nat, 3)`.
        """
        if use_functorch is True:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            chi = ...
            assert isinstance(chi, Tensor)
        else:
            a = self.polarizability(numbers, positions, chrg, spin)  # (3, 3)

            # d(3, 3) / d(nat, 3) -> (3, 3, nat*3) -> (3, 3, nat, 3)
            chi = _jac(a, positions).reshape((3, 3, *positions.shape[-2:]))

        return chi

    def pol_deriv_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = 1.0e-5,
    ) -> Tensor:
        # turn off printing in numerical calcs
        OutputHandler.temporary_disable_on()
        logger.debug("Polarizability numerical derivative: Start.")

        # pylint: disable=import-outside-toplevel
        import gc

        # important: use new/separate position tensor
        pos = positions.detach().clone()

        # (3, 3, nat, 3)
        deriv = torch.zeros((3, 3, *positions.shape[-2:]), **self.dd)

        logger.debug(
            "Polarizability numerical derivative: Start building numerical "
            f"dipole derivative {deriv.shape})."
        )

        count = 1
        nsteps = 3 * numbers.shape[-1]
        for i in range(numbers.shape[-1]):
            for j in range(3):
                pos[i, j] += step_size
                r = self.polarizability(numbers, pos, chrg, spin)

                pos[i, j] -= 2 * step_size
                l = self.polarizability(numbers, pos, chrg, spin)

                pos[i, j] += step_size
                deriv[:, :, i, j] = 0.5 * (r - l) / step_size

                logger.debug(
                    "Polarizability numerical derivative: Step " f"{count}/{nsteps}"
                )
                count += 1

            gc.collect()
        gc.collect()

        logger.debug("Polarizability numerical derivative: All finished.")
        OutputHandler.temporary_disable_off()

        return deriv

    @requires_efield
    @requires_efield_grad
    def hyperpol(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
        derived_quantity: Literal["energy", "dipole", "pol"] = "pol",
    ) -> Tensor:
        r"""
        Calculate the hyper polarizability tensor :math:`\beta`.

        .. math::

            \beta = \dfrac{\partial \alpha}{\partial F} = \dfrac{\partial^2 \mu}{\partial F^2} = \dfrac{\partial^3 E}{\partial^2 3}

        One can calculate the Jacobian either row-by-row using the standard
        `torch.autograd.grad` with unit vectors in the VJP (see `here`_) or
        using `torch.func`'s function transforms (e.g., `jacrev`).

        .. _here: https://pytorch.org/functorch/stable/notebooks/jacobians_hessians.html#computing-the-jacobian

        Note
        ----
        Using `torch.func`'s function transforms can apparently be only used
        once. Hence, for example, the Hessian and the polarizability cannot
        be both calculated with functorch.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.
        derived_quantity: Literal['energy', 'dipole'], optional
            Which derivative to calculate for the polarizability, i.e.,
            derivative of dipole moment or energy w.r.t field.

        Returns
        -------
        Tensor
            Hyper polarizability tensor of shape `(3, 3, 3)`.
        """
        # retrieve the efield interaction and the field
        field = self.interactions.get_interaction(efield.LABEL_EFIELD).field

        if use_functorch is False:
            alpha = self.polarizability(
                numbers, positions, chrg, spin, use_functorch=use_functorch
            )
            return _jac(alpha, field)

        # pylint: disable=import-outside-toplevel
        from tad_mctc.autograd import jacrev

        if derived_quantity == "pol":

            def wrapped_polarizability(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return self.polarizability(numbers, positions, chrg, spin)

            beta = jacrev(wrapped_polarizability)(field)

        elif derived_quantity == "dipole":

            def wrapped_dipole(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return self.dipole(numbers, positions, chrg, spin)

            beta = jacrev(jacrev(wrapped_dipole))(field)

        elif derived_quantity == "energy":

            def wrapped_energy(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return self.energy(numbers, positions, chrg, spin)

            beta = jacrev(jacrev(jacrev(wrapped_energy)))(field)

        else:
            raise ValueError(
                f"Unknown `derived_quantity` '{derived_quantity}'. The "
                "polarizability can be calculated as the derivative of the "
                "'dipole' moment or the 'energy'."
            )

        # 3x3x3 hyper polarizability tensor
        assert isinstance(beta, Tensor)
        return beta

    @requires_efield
    def hyperpol_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = 1.0e-5,
    ) -> Tensor:
        """
        Numerical polarizability.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        chrg : Tensor | float | int | str | None
            Total charge. Defaults to `None`.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        step_size : float | int, optional
            Step size for the numerical derivative.

        Returns
        -------
        Tensor
            Polarizability tensor.
        """
        # retrieve the efield interaction and the field and detach for gradient
        ef = self.interactions.get_interaction(efield.LABEL_EFIELD)
        field = ef.field.detach().clone()
        self.interactions.update_efield(field=field)

        # pylint: disable=import-outside-toplevel
        import gc

        deriv = torch.zeros(*(*numbers.shape[:-1], 3, 3, 3), **self.dd)

        # turn off printing in numerical hessian
        OutputHandler.temporary_disable_on()
        logger.debug(f"Numerical Hyper Polarizability: Starting build ({deriv.shape})")

        count = 1
        for i in range(3):
            field[..., i] += step_size
            self.interactions.update_efield(field=field)
            gr = self.polarizability_numerical(numbers, positions, chrg, spin)

            field[..., i] -= 2 * step_size
            self.interactions.update_efield(field=field)
            gl = self.polarizability_numerical(numbers, positions, chrg, spin)

            field[..., i] += step_size
            self.interactions.update_efield(field=field)
            deriv[..., :, :, i] = 0.5 * (gr - gl) / step_size

            logger.debug(f"Numerical Hyper Polarizability: step {count}/{3}")
            count += 1

            gc.collect()

        logger.debug("Numerical Hyper Polarizability: All finished.")
        OutputHandler.temporary_disable_off()

        return deriv

    # SPECTRA

    def ir(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
    ) -> tuple[Tensor, Tensor]:
        logger.debug("IR spectrum: Start.")

        # run vibrational analysis first
        hess = self.hessian(numbers, positions, chrg, spin)
        freqs, modes = properties.frequencies(numbers, positions, hess)

        self.integrals.invalidate_driver()

        dmu_dr = self.dipole_deriv(numbers, positions, chrg, spin)  # (3, nat, 3)
        dmu_dr = dmu_dr.view(3, -1)  # (3, nat*3)

        # (3, nat*3) @ (nat*3, nfreqs) = (3, nfreqs)
        dmu_dq = torch.matmul(dmu_dr, modes)

        ir_ints = torch.einsum("...df,...df->...f", dmu_dq, dmu_dq)  # (nfreqs,)

        # TODO: Figure out unit
        from ..constants import units

        logger.debug("IR spectrum numerical: All finished.")

        return freqs * units.AU2RCM, ir_ints * 1378999.7790799031

    def ir_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = 1.0e-5,
    ) -> tuple[Tensor, Tensor]:
        # turn off printing in numerical calcs
        OutputHandler.temporary_disable_on()
        logger.debug("IR spectrum numerical: Start.")

        # run vibrational analysis first
        hess = self.hessian_numerical(numbers, positions, chrg)
        freqs, modes = properties.frequencies(numbers, positions, hess)

        # calculate nulcear dipole derivative dmu/dR (3, nat, 3)
        dmu_dr = self.dipole_deriv_numerical(
            numbers, positions, chrg, spin, step_size=step_size
        )
        dmu_dr = dmu_dr.view(3, -1)  # (3, nat*3)

        dmu_dq = torch.matmul(dmu_dr, modes)  # (3, nfreqs)
        ir_ints = torch.einsum("...df,...df->...f", dmu_dq, dmu_dq)  # (nfreqs,)

        # TODO: Figure out unit
        from ..constants import units

        logger.debug("IR spectrum numerical: All finished.")
        OutputHandler.temporary_disable_off()

        return freqs * units.AU2RCM, ir_ints * 1378999.7790799031

    def raman(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the frequency and static intensities of Raman spectra.
        Formula taken from `here <https://doi.org/10.1080/00268970701516412>`__.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        positions : Tensor
            Atomic positions of shape (n, 3).
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        chrg : Tensor | float | int | str | None
            Total charge. Defaults to `None`.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.

        Returns
        -------
        tuple[Tensor, Tensor]
            Raman frequencies and intensities.

        Raises
        ------
        RuntimeError
            `positions` tensor does not have `requires_grad=True`.
        """
        freqs, modes = self.vibration(numbers, positions, chrg, spin)

        # d(3, 3) / d(nat, 3) -> (3, 3, nat, 3) -> (3, 3, nat*3)
        da_dr = self.pol_deriv(numbers, positions, chrg, spin)
        da_dr = da_dr.reshape((3, 3, -1))

        # (3, 3, nat*3) * (nat*3, nmodes) = (3, 3, nmodes)
        da_dq = torch.matmul(da_dr, modes)

        # Eq.3 with alpha' = a
        a = torch.einsum("...iij->...j", da_dq)

        # Eq.4 with (gamma')^2 = g = 0.5 * (g1 + g2 + g3 + 6.0*g4)
        g1 = (da_dq[0, 0] - da_dq[1, 1]) ** 2
        g2 = (da_dq[0, 0] - da_dq[2, 2]) ** 2
        g3 = (da_dq[2, 2] - da_dq[1, 1]) ** 2
        g4 = da_dq[0, 1] ** 2 + da_dq[1, 2] ** 2 + da_dq[2, 0] ** 2
        g = g1 + g2 + g3 + 6.0 * g4

        # Eq.1 (the 1/3 from Eq.3 is squared, yielding 45 * 1/9 = 5; the 7 is
        # halfed by the 0.5 from Eq.4)
        raman_ints = 5 * torch.pow(a, 2.0) + 3.5 * g

        return freqs, raman_ints

    def raman_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
    ) -> tuple[Tensor, Tensor]:
        freqs, modes = self.vibration_numerical(numbers, positions, chrg, spin)

        # d(3, 3) / d(nat, 3) -> (3, 3, nat, 3) -> (3, 3, nat*3)
        da_dr = self.pol_deriv_numerical(numbers, positions, chrg, spin)
        da_dr = da_dr.reshape((3, 3, -1))

        # (3, 3, nat*3) * (nat*3, nmodes) = # (3, 3, nmodes)
        da_dq = torch.matmul(da_dr, modes)

        # Eq.3 with alpha' = a
        a = torch.einsum("...iij->...j", da_dq)

        # Eq.4 with (gamma')^2 = g = 0.5 * (g1 + g2 + g3 + g4)
        g1 = (da_dq[0, 0] - da_dq[1, 1]) ** 2
        g2 = (da_dq[0, 0] - da_dq[2, 2]) ** 2
        g3 = (da_dq[2, 2] - da_dq[1, 1]) ** 2
        g4 = da_dq[0, 1] ** 2 + da_dq[1, 2] ** 2 + da_dq[2, 0] ** 2
        g = 0.5 * (g1 + g2 + g3 + g4)

        # Eq.1 (the 1/3 from Eq.3 is squared and reduces the 45)
        raman_ints = 5 * torch.pow(a, 2.0) + 7 * g

        return freqs, raman_ints
