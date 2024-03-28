"""
Base calculator for the extended tight-binding model.
"""

from __future__ import annotations

import logging

import torch
from tad_mctc.convert import any_to_tensor
from tad_mctc.exceptions import DtypeError
from tad_mctc.typing import Any, Literal, Sequence, Tensor, TensorLike

from .. import integral as ints
from .. import scf
from ..basis import IndexHelper
from ..components.classicals import (
    Classical,
    ClassicalList,
    Dispersion,
    Halogen,
    Repulsion,
    new_dispersion,
    new_halogen,
    new_repulsion,
)
from ..components.interactions import Charges, Interaction, InteractionList, Potential
from ..components.interactions.coulomb import new_es2, new_es3
from ..components.interactions.external import field as efield
from ..components.interactions.external import fieldgrad as efield_grad
from ..config import Config
from ..constants import defaults
from ..io import OutputHandler
from ..param import Param
from ..properties import moments
from ..properties import vibration as vib
from ..timing import timer
from ..utils import _jac, einsum
from ..wavefunction import filling
from . import decorators as cdec

__all__ = ["Calculator", "Result"]


logger = logging.getLogger(__name__)


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
        self.cenergies = {}

    def __str__(self) -> str:
        """Custom print representation showing all available slots."""
        return f"{self.__class__.__name__}({self.__slots__})"

    def __repr__(self) -> str:
        """Custom print representation showing all available slots."""
        return str(self)

    def get_energies(self) -> dict[str, dict[str, Any]]:
        """
        Get energies in a dictionary.

        Returns
        -------
        dict[str, dict[str, float]]
            Energies in a dictionary.
        """
        KEY = "value"

        c = {k: {KEY: v.sum().item()} for k, v in self.cenergies.items()}
        ctotal = sum(d[KEY] for d in c.values())

        e = {
            "SCF": {KEY: self.scf.sum().item()},
            "Free Energy (Fermi)": {KEY: self.fenergy.sum().item()},
        }
        etotal = sum(d[KEY] for d in e.values())

        return {
            "total": {KEY: self.total.sum().item()},
            "Classical": {KEY: ctotal, "sub": c},
            "Electronic": {KEY: etotal, "sub": e},
        }

    def print_energies(
        self, v: int = 4, precision: int = 14
    ) -> None:  # pragma: no cover
        """Print energies in a table."""

        # pylint: disable=import-outside-toplevel
        from ..io import OutputHandler

        OutputHandler.write_table(
            self.get_energies(),
            title="Energies",
            columns=["Contribution", "Energy (Eh)"],
            v=v,
            precision=precision,
        )


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

        __slots__ = ["_disabled", "energy", "forces", "hessian"]

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
            self.hessian = None

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
            if key == "wrapper":
                raise RuntimeError(
                    "Key 'wrapper' detected. This happens if the cache "
                    "decorator is not the innermost decorator of the "
                    "Calculator method that you are trying to cache. Please "
                    "move the cache decorator to the innermost position. "
                    "Otherwise, the name of the method cannot be inferred "
                    "correctly."
                )

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
        timer.start("Calculator", parent_uid="Setup")

        # setup verbosity first
        opts = opts if opts is not None else {}
        if isinstance(opts, dict):
            OutputHandler.verbosity = opts.pop("verbosity", None)

        OutputHandler.write_stdout("", v=5)
        OutputHandler.write_stdout("", v=5)
        OutputHandler.write_stdout("===========", v=4)
        OutputHandler.write_stdout("CALCULATION", v=4)
        OutputHandler.write_stdout("===========", v=4)
        OutputHandler.write_stdout("", v=4)
        OutputHandler.write_stdout("Setup Calculator", v=4)

        allowed_dtypes = (torch.long, torch.int16, torch.int32, torch.int64)
        if numbers.dtype not in allowed_dtypes:
            raise DtypeError(
                "Dtype of atomic numbers must be one of the following to allow "
                f"indexing: '{', '.join([str(x) for x in allowed_dtypes])}', "
                f"but is '{numbers.dtype}'"
            )

        super().__init__(device, dtype)
        dd = {"device": self.device, "dtype": self.dtype}

        # setup calculator options
        if isinstance(opts, dict):
            opts = Config(**opts, **dd)
        self.opts = opts

        # create cache
        self.cache = self.Cache(**dd)

        self.batched = numbers.ndim > 1

        self.ihelp = IndexHelper.from_numbers(numbers, par)

        ################
        # INTERACTIONS #
        ################

        # setup self-consistent contributions
        OutputHandler.write_stdout_nf(" - Interactions      ... ", v=4)

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

        OutputHandler.write_stdout("done", v=4)

        ##############
        # CLASSICALS #
        ##############

        # setup non-self-consistent contributions
        OutputHandler.write_stdout_nf(" - Classicals        ... ", v=4)

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

        OutputHandler.write_stdout("done", v=4)

        #############
        # INTEGRALS #
        #############

        OutputHandler.write_stdout_nf(" - Integrals         ... ", v=4)

        # figure out integral level from interactions
        if efield.LABEL_EFIELD in self.interactions.labels:
            if self.opts.ints.level < ints.INTLEVEL_DIPOLE:
                OutputHandler.warn(
                    "Setting integral level to DIPOLE "
                    f"({ints.INTLEVEL_DIPOLE}) due to electric field "
                    "interaction."
                )
            self.opts.ints.level = max(ints.INTLEVEL_DIPOLE, self.opts.ints.level)
        if efield_grad.LABEL_EFIELD_GRAD in self.interactions.labels:
            if self.opts.ints.level < ints.INTLEVEL_DIPOLE:
                OutputHandler.warn(
                    "Setting integral level to QUADRUPOLE "
                    f"{ints.INTLEVEL_DIPOLE} due to electric field gradient "
                    "interaction."
                )
            self.opts.ints.level = max(ints.INTLEVEL_QUADRUPOLE, self.opts.ints.level)

        # setup integral
        driver = self.opts.ints.driver
        self.integrals = ints.Integrals(numbers, par, self.ihelp, driver=driver, **dd)

        if self.opts.ints.level >= ints.INTLEVEL_OVERLAP:
            self.integrals.hcore = ints.Hamiltonian(numbers, par, self.ihelp, **dd)
            self.integrals.overlap = ints.Overlap(driver=driver, **dd)

        if self.opts.ints.level >= ints.INTLEVEL_DIPOLE:
            self.integrals.dipole = ints.Dipole(driver=driver, **dd)

        if self.opts.ints.level >= ints.INTLEVEL_QUADRUPOLE:
            self.integrals.quadrupole = ints.Quadrupole(driver=driver, **dd)

        OutputHandler.write_stdout("done\n", v=4)

        timer.stop("Calculator")

    def reset(self) -> None:
        self.classicals.reset_all()
        self.interactions.reset_all()
        self.integrals.reset_all()

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
        OutputHandler.write_stdout("Singlepoint ", v=3)

        chrg = any_to_tensor(chrg, **self.dd)
        if spin is not None:
            spin = any_to_tensor(spin, **self.dd)

        result = Result(positions, device=self.device, dtype=self.dtype)

        # CLASSICAL CONTRIBUTIONS

        if len(self.classicals.components) > 0:
            OutputHandler.write_stdout_nf(" - Classicals        ... ", v=3)
            timer.start("Classicals")

            ccaches = self.classicals.get_cache(numbers, self.ihelp)
            cenergies = self.classicals.get_energy(positions, ccaches)
            result.cenergies = cenergies
            result.total += torch.stack(list(cenergies.values())).sum(0)

            timer.stop("Classicals")
            OutputHandler.write_stdout("done", v=3)

            if grad is True:
                cgradients = self.classicals.get_gradient(cenergies, positions)
                result.cgradients = cgradients
                result.total_grad += torch.stack(list(cgradients.values())).sum(0)

        # SELF-CONSISTENT FIELD PROCEDURE
        if not any(x in ["all", "scf"] for x in self.opts.exclude):
            timer.start("Integrals")
            # overlap integral
            OutputHandler.write_stdout_nf(" - Overlap           ... ", v=3)
            timer.start("Overlap", parent_uid="Integrals")
            self.integrals.build_overlap(positions)
            timer.stop("Overlap")
            OutputHandler.write_stdout("done", v=3)

            # Core Hamiltonian integral (requires overlap internally!)
            OutputHandler.write_stdout_nf(" - Core Hamiltonian  ... ", v=3)
            timer.start("Core Hamiltonian", parent_uid="Integrals")
            self.integrals.build_hcore(positions)
            timer.stop("Core Hamiltonian")
            OutputHandler.write_stdout("done", v=3)

            # dipole integral
            if self.opts.ints.level >= ints.INTLEVEL_DIPOLE:
                OutputHandler.write_stdout_nf(" - Dipole            ... ", v=3)
                timer.start("Dipole Integral", parent_uid="Integrals")
                self.integrals.build_dipole(positions)
                timer.stop("Dipole Integral")
                OutputHandler.write_stdout("done", v=3)

            # quadrupole integral
            if self.opts.ints.level >= ints.INTLEVEL_QUADRUPOLE:
                OutputHandler.write_stdout_nf(" - Quadrupole        ... ", v=3)
                timer.start("Quadrupole Integral", parent_uid="Integrals")
                self.integrals.build_quadrupole(positions)
                timer.stop("Quadrupole Integral")
                OutputHandler.write_stdout("done", v=3)

            timer.stop("Integrals")

            # TODO: Think about handling this case
            if self.integrals.hcore is None:
                raise RuntimeError
            if self.integrals.overlap is None:
                raise RuntimeError

            timer.start("SCF", "Self-Consistent Field")

            # get caches of all interactions
            timer.start("Interaction Cache", parent_uid="SCF")
            OutputHandler.write_stdout_nf(" - Interaction Cache ... ", v=3)
            icaches = self.interactions.get_cache(
                numbers=numbers, positions=positions, ihelp=self.ihelp
            )
            timer.stop("Interaction Cache")
            OutputHandler.write_stdout("done", v=3)

            # SCF
            OutputHandler.write_stdout("\nStarting SCF Iterations...", v=3)

            scf_results = scf.solve(
                numbers,
                positions,
                chrg,
                spin,
                self.interactions,
                icaches,
                self.ihelp,
                self.opts.scf,
                self.integrals.matrices,
                self.integrals.hcore.integral.refocc,
            )

            timer.stop("SCF")
            OutputHandler.write_stdout(
                f"SCF converged in {scf_results['iterations']} iterations.", v=3
            )

            # store SCF results
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

            if not self.batched:
                OutputHandler.write_stdout(
                    f"SCF Energy  : {result.scf.sum(-1):.14f} Hartree.",
                    v=2,
                )
                OutputHandler.write_stdout(
                    f"Total Energy: {result.total.sum(-1):.14f} Hartree.", v=1
                )

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

                from .. import ncoord

                cn = ncoord.get_coordination_number(numbers, positions)
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

    @cdec.cache
    def energy(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
    ) -> Tensor:
        return self.singlepoint(numbers, positions, chrg, spin).total.sum(-1)

    @cdec.requires_positions_grad
    @cdec.cache
    def forces(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
    ) -> Tensor:
        r"""
        Calculate the electric dipole moment :math:`f` via AD.

        .. math::

            f = -\dfrac{\partial E}{\partial R}

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
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.

        Returns
        -------
        Tensor
            Atomic forces of shape `(..., nat, 3)`.
        """
        logger.debug("Forces: Starting.")

        if use_functorch is True:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            # jacrev requires a scalar from `self.energy`!
            jac_func = jacrev(self.energy, argnums=1)
            jac = jac_func(numbers, positions, chrg, spin)
            assert isinstance(jac, Tensor)
        else:
            energy = self.energy(numbers, positions, chrg, spin)
            jac = _jac(energy, positions)

        if jac.is_contiguous() is False:
            logger.debug(
                "Hessian: Re-enforcing contiguous memory layout after "
                f"autodiff (use_functorch={use_functorch})."
            )
            jac = jac.contiguous()

        logger.debug("Forces: All finished.")
        return -jac

    @cdec.numerical
    @cdec.cache
    def forces_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
    ) -> Tensor:
        r"""
        Numerically calculate the atomic forces :math:`f`.

        .. math::

            f = -\dfrac{\partial E}{\partial R}

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        step_size : int | float, optional
            Step size for numerical differentiation.

        Returns
        -------
        Tensor
            Atomic forces of shape `(..., nat, 3)`.
        """
        # pylint: disable=import-outside-toplevel
        import gc

        # (..., nat, 3)
        deriv = torch.zeros(positions.shape, **self.dd)
        logger.debug(f"Forces (numerical): Starting build ({deriv.shape}).")

        count = 1
        nsteps = 3 * numbers.shape[-1]
        for i in range(numbers.shape[-1]):
            for j in range(3):
                positions[..., i, j] += step_size
                gr = self.energy(numbers, positions, chrg, spin)

                positions[..., i, j] -= 2 * step_size
                gl = self.energy(numbers, positions, chrg, spin)

                positions[..., i, j] += step_size
                deriv[..., i, j] = 0.5 * (gr - gl) / step_size

                logger.debug(f"Forces (numerical): step {count}/{nsteps}")
                count += 1

                gc.collect()
            gc.collect()

        logger.debug("Forces (numerical): All finished.")

        return -deriv

    @cdec.requires_positions_grad
    @cdec.cache
    def hessian(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
        matrix: bool = False,
    ) -> Tensor:
        """
        Calculation of the nuclear Hessian with AD.

        Note
        ----
        The `jacrev` function of `functorch` requires scalars for the expected
        behavior, i.e., the nuclear Hessian only acquires the expected shape of
        `(nat, 3, nat, 3)` if the energy is provided as a scalar value.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_functorch : bool, optional
            Whether to use `functorch` for autodiff. Defaults to `False`.
        matrix : bool, optional
            Whether to reshape the Hessian to a matrix, i.e., `(nat*3, nat*3)`.
            Defaults to `False`.

        Returns
        -------
        Tensor
            Hessian of shape `(..., nat, 3, nat, 3)` or `(..., nat*3, nat*3)`.

        Raises
        ------
        RuntimeError
            Positions tensor does not have `requires_grad=True`.
        """
        logger.debug("Autodiff Hessian: Starting Calculation.")

        if use_functorch is True:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            # jacrev requires a scalar from `self.energy`!
            hess_func = jacrev(jacrev(self.energy, argnums=1), argnums=1)
            hess = hess_func(numbers, positions, chrg, spin)
            assert isinstance(hess, Tensor)
        else:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import hessian

            # jacrev requires a scalar from `self.energy`!
            hess = hessian(self.energy, (numbers, positions, chrg, spin), argnums=1)

        if hess.is_contiguous() is False:
            logger.debug(
                "Hessian: Re-enforcing contiguous memory layout after "
                f"autodiff (use_functorch={use_functorch})."
            )
            hess = hess.contiguous()

        # reshape (..., nat, 3, nat, 3) to (..., nat*3, nat*3)
        if matrix is True:
            s = [*numbers.shape[:-1], *2 * [3 * numbers.shape[-1]]]
            hess = hess.view(*s)

        self.cache["hessian"] = hess

        logger.debug("Autodiff Hessian: All finished.")

        return hess

    @cdec.numerical
    @cdec.cache
    def hessian_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
        matrix: bool = False,
    ) -> Tensor:
        """
        Numerically calculate the Hessian.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int | str | None
            Total charge. Defaults to `None`.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        step_size : int | float, optional
            Step size for numerical differentiation.
        matrix : bool, optional
            Whether to reshape the Hessian to a matrix, i.e., (nat*3, nat*3).
            Defaults to `False`.

        Returns
        -------
        Tensor
            Hessian of shape `(..., nat, 3, nat, 3) or `(..., nat*3, nat*3)`.
        """
        # pylint: disable=import-outside-toplevel
        import gc

        def _gradfcn(pos: Tensor) -> Tensor:
            with torch.enable_grad():
                pos.requires_grad_(True)
                result = self.singlepoint(numbers, pos, chrg, spin, grad=True)
                pos.detach_()
            return result.total_grad.detach()

        # (..., nat, 3, nat, 3)
        deriv = torch.zeros((*positions.shape, *positions.shape[-2:]), **self.dd)
        logger.debug(f"Hessian (numerical): Starting build ({deriv.shape}).")

        count = 1
        nsteps = 3 * numbers.shape[-1]
        for i in range(numbers.shape[-1]):
            for j in range(3):
                positions[..., i, j] += step_size
                gr = _gradfcn(positions)

                positions[..., i, j] -= 2 * step_size
                gl = _gradfcn(positions)

                positions[..., i, j] += step_size
                deriv[..., :, :, i, j] = 0.5 * (gr - gl) / step_size

                logger.debug(f"Hessian (numerical): step {count}/{nsteps}")
                count += 1

                gc.collect()
            gc.collect()

        # reshape (..., nat, 3, nat, 3) to (..., nat*3, nat*3)
        if matrix is True:
            s = [*numbers.shape[:-1], *2 * [3 * numbers.shape[-1]]]
            deriv = deriv.reshape(*s)

        logger.debug("Hessian (numerical): All finished.")

        return deriv

    def vibration(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
        project_translational: bool = True,
        project_rotational: bool = True,
    ) -> vib.VibResult:
        r"""
        Perform vibrational analysis. This calculates the Hessian matrix and
        diagonalizes it to obtain the vibrational frequencies and normal modes.

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
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.
        project_translational : bool, optional
            Project out translational modes. Defaults to `True`.
        project_rotational : bool, optional
            Project out rotational modes. Defaults to `True`.

        Returns
        -------
        vib.VibResult
            Result container with vibrational frequencies (shape:
            `(..., nfreqs)`) and normal modes (shape: `(..., nat*3, nfreqs)`).
        """
        hess = self.hessian(
            numbers,
            positions,
            chrg,
            spin,
            use_functorch=use_functorch,
            matrix=False,
        )
        return vib.vib_analysis(
            numbers,
            positions,
            hess,
            project_translational=project_translational,
            project_rotational=project_rotational,
        )

    @cdec.numerical
    def vibration_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
        project_translational: bool = True,
        project_rotational: bool = True,
    ) -> vib.VibResult:
        r"""
        Perform vibrational analysis via numerical Hessian.
        The Hessian matrix is diagonalized to obtain the vibrational
        frequencies and normal modes.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        step_size : int | float, optional
            Step size for numerical differentiation.
        project_translational : bool, optional
            Project out translational modes. Defaults to `True`.
        project_rotational : bool, optional
            Project out rotational modes. Defaults to `True`.

        Returns
        -------
        vib.VibResult
            Result container with vibrational frequencies (shape:
            `(..., nfreqs)`) and normal modes (shape: `(..., nat*3, nfreqs)`).
        """
        hess = self.hessian_numerical(
            numbers, positions, chrg, spin, step_size=step_size
        )
        return vib.vib_analysis(
            numbers,
            positions,
            hess,
            project_translational=project_translational,
            project_rotational=project_rotational,
        )

    @cdec.requires_efield
    @cdec.requires_efield_grad
    def dipole(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
    ) -> Tensor:
        r"""
        Calculate the electric dipole moment :math:`\mu` via AD.

        .. math::

            \mu = \dfrac{\partial E}{\partial F}

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
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.

        Returns
        -------
        Tensor
            Electric dipole moment of shape `(..., 3)`.
        """
        field = self.interactions.get_interaction(efield.LABEL_EFIELD).field

        if use_functorch is True:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            def wrapped_energy(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return self.energy(numbers, positions, chrg, spin)

            dip = jacrev(wrapped_energy)(field)
            assert isinstance(dip, Tensor)
        else:
            # calculate electric dipole contribution from xtb energy: -de/dE
            energy = self.energy(numbers, positions, chrg, spin)
            dip = _jac(energy, field)

        if dip.is_contiguous() is False:
            logger.debug(
                "Dipole moment: Re-enforcing contiguous memory layout "
                f"after autodiff (use_functorch={use_functorch})."
            )
            dip = dip.contiguous()

        return -dip

    @cdec.requires_efield
    def dipole_analytical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        *_,  # absorb stuff
    ) -> Tensor:
        r"""
        Analytically calculate the electric dipole moment :math:`\mu`.

        .. math::

            \mu = \dfrac{\partial E}{\partial F}

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.

        Returns
        -------
        Tensor
            Electric dipole moment of shape `(..., 3)`.
        """
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
        dip = moments.dipole(qat, positions, result.density, dipint.matrix)
        return dip

    @cdec.numerical
    @cdec.requires_efield
    def dipole_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
    ) -> Tensor:
        r"""
        Numerically calculate the electric dipole moment :math:`\mu`.

        .. math::

            \mu = \dfrac{\partial E}{\partial F}

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        step_size : int | float, optional
            Step size for numerical differentiation.

        Returns
        -------
        Tensor
            Electric dipole moment of shape `(..., 3)`.
        """
        # pylint: disable=import-outside-toplevel
        import gc

        # retrieve electric field, no copy needed because of no_grad context
        field = self.interactions.get_interaction(efield.LABEL_EFIELD).field

        # (..., 3)
        deriv = torch.zeros((*numbers.shape[:-1], 3), **self.dd)
        logger.debug(f"Dipole (numerical): Starting build ({deriv.shape}).")

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

            logger.debug(f"Dipole (numerical): step {count}/{3}")
            count += 1

            gc.collect()

        logger.debug("Dipole (numerical): All finished.")

        return -deriv

    @cdec.requires_positions_grad
    def dipole_deriv(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_analytical: bool = True,
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
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_analytical: bool, optional
            Whether to use the analytically calculated dipole moment for AD or
            the automatically differentiated dipole moment.
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.

        Returns
        -------
        Tensor
            Cartesian dipole derivative of shape `(..., 3, nat, 3)`.
        """

        if use_analytical is True:
            dip_fcn = self.dipole_analytical
        else:
            dip_fcn = self.dipole

        if use_functorch is True:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            # d(3) / d(nat, 3) = (3, nat, 3)
            dmu_dr = jacrev(dip_fcn, argnums=1)(
                numbers, positions, chrg, spin, use_functorch
            )
            assert isinstance(dmu_dr, Tensor)

        else:
            mu = dip_fcn(numbers, positions, chrg, spin, use_functorch)

            # (..., 3, 3*nat) -> (..., 3, nat, 3)
            dmu_dr = _jac(mu, positions).reshape(
                (*numbers.shape[:-1], 3, *positions.shape[-2:])
            )

        if dmu_dr.is_contiguous() is False:
            logger.debug(
                "Dipole derivative: Re-enforcing contiguous memory layout "
                f"after autodiff (use_functorch={use_functorch})."
            )
            dmu_dr = dmu_dr.contiguous()

        return dmu_dr

    @cdec.numerical
    def dipole_deriv_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
    ) -> Tensor:
        r"""
        Numerically calculate cartesian dipole derivative :math:`\mu'`.

        .. math::

            \mu' = \dfrac{\partial \mu}{\partial R} = \dfrac{\partial^2 E}{\partial F \partial R}

        Here, the analytical dipole moment is used for the numerical
        differentiation.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        step_size: int | float, optional
            Step size for numerical differentiation.

        Returns
        -------
        Tensor
            Cartesian dipole derivative of shape `(..., 3, nat, 3)`.
        """
        # pylint: disable=import-outside-toplevel
        import gc

        # (..., 3, n, 3)
        deriv = torch.zeros(
            (*numbers.shape[:-1], 3, *positions.shape[-2:]),
            **self.dd,
        )
        logger.debug(f"Dipole derivative (numerical): Starting build ({deriv.shape}).")

        count = 1
        nsteps = 3 * numbers.shape[-1]

        for i in range(numbers.shape[-1]):
            for j in range(3):
                positions[..., i, j] += step_size
                r = self.dipole_analytical(numbers, positions, chrg, spin)

                positions[..., i, j] -= 2 * step_size
                l = self.dipole_analytical(numbers, positions, chrg, spin)

                positions[..., i, j] += step_size
                deriv[..., :, i, j] = 0.5 * (r - l) / step_size

                logger.debug("Dipole derivative (numerical): Step " f"{count}/{nsteps}")
                count += 1

                gc.collect()
            gc.collect()

        logger.debug("Dipole derivative (numerical): All finished.")

        return deriv

    @cdec.requires_efg
    @cdec.requires_efg_grad
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
                return self.energy(numbers, positions, chrg, spin)

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

        print("\nquad_moment\n", e_quad)

        e_quad = e_quad.view(3, 3)

        print("quad_moment", e_quad.shape)

        cart = torch.empty((6), **self.dd)

        tr = 0.5 * einsum("...ii->...", e_quad)
        print("tr", tr)
        cart[..., 0] = 1.5 * e_quad[..., 0, 0] - tr
        cart[..., 1] = 3.0 * e_quad[..., 1, 0]
        cart[..., 2] = 1.5 * e_quad[..., 1, 1] - tr
        cart[..., 3] = 3.0 * e_quad[..., 2, 0]
        cart[..., 4] = 3.0 * e_quad[..., 2, 1]
        cart[..., 5] = 1.5 * e_quad[..., 2, 2] - tr

        print("cart\n", cart)

        # electric quadrupole contribution form nuclei: sum_i(r_ik * Z_i)
        n_quad = einsum(
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

    @cdec.requires_efg
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

        return moments.quadrupole(qat, dpat, qpat, positions)

    @cdec.numerical
    @cdec.requires_efg
    def quadrupole_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
    ) -> Tensor:
        # retrieve the efg interaction and the field gradient and detach
        efg = self.interactions.get_interaction(efield_grad.LABEL_EFIELD_GRAD)
        _field_grad = efg.field_grad.clone()
        field_grad = efg.field_grad.detach().clone()
        self.interactions.update_efield_grad(field_grad=field_grad)

        # (..., 3, 3)
        deriv = torch.zeros((*numbers.shape[:-1], 3, 3), **self.dd)
        logger.debug(f"Quadrupole (numerical): Starting build ({deriv.shape}).")

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

                logger.debug(f"Quadrupole (numerical): step {count}/{3}")
                count += 1

        # reset
        self.interactions.update_efield_grad(field_grad=_field_grad)

        logger.debug("Quadrupole (numerical): All finished.")

        return deriv

    @cdec.requires_efield
    @cdec.requires_efield_grad
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
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
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
            Polarizability tensor of shape `(..., 3, 3)`.
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

            # negative sign already in dipole but not in energy derivative!
            alpha = -alpha
        else:
            raise ValueError(
                f"Unknown `derived_quantity` '{derived_quantity}'. The "
                "polarizability can be calculated as the derivative of the "
                "'dipole' moment or the 'energy'."
            )

        if alpha.is_contiguous() is False:
            logger.debug(
                "Polarizability: Re-enforcing contiguous memory layout "
                f"after autodiff (use_functorch={use_functorch})."
            )
            alpha = alpha.contiguous()

        # 3x3 polarizability tensor
        return alpha

    @cdec.numerical
    @cdec.requires_efield
    def polarizability_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
    ) -> Tensor:
        r"""
        Numerically calculate the polarizability tensor :math:`\alpha`.

        .. math::

            \alpha = \dfrac{\partial \mu}{\partial F} = \dfrac{\partial^2 E}{\partial^2 F}

        Here, the analytical dipole moment is used for the numerical derivative.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int | str | None
            Total charge. Defaults to `None`.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        step_size : float | int, optional
            Step size for the numerical derivative.

        Returns
        -------
        Tensor
            Polarizability tensor of shape `(..., 3, 3)`.
        """
        # pylint: disable=import-outside-toplevel
        import gc

        # retrieve the efield interaction and the field and detach for gradient
        ef = self.interactions.get_interaction(efield.LABEL_EFIELD)
        _field = ef.field.clone()
        field = ef.field.detach().clone()
        self.interactions.update_efield(field=field)

        # (..., 3, 3)
        deriv = torch.zeros(*(*numbers.shape[:-1], 3, 3), **self.dd)
        logger.debug(f"Polarizability (numerical): Starting build ({deriv.shape})")

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

            logger.debug(f"Polarizability (numerical): step {count}/{3}")
            count += 1

            gc.collect()

        logger.debug("Polarizability (numerical): All finished.")

        # explicitly update field (to restore original field with possible grad)
        self.interactions.reset_efield()
        self.interactions.update_efield(field=_field)

        return deriv

    @cdec.requires_efield
    @cdec.requires_positions_grad
    def pol_deriv(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
        derived_quantity: Literal["energy", "dipole", "pol"] = "pol",
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
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
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
            Polarizability derivative shape `(..., 3, 3, nat, 3)`.
        """
        if use_functorch is False:
            a = self.polarizability(
                numbers, positions, chrg, spin, use_functorch=use_functorch
            )

            # d(3, 3) / d(nat, 3) -> (3, 3, nat*3) -> (3, 3, nat, 3)
            chi = _jac(a, positions).reshape((3, 3, *positions.shape[-2:]))

        else:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            chi = jacrev(self.polarizability, argnums=1)(
                numbers, positions, chrg, spin, use_functorch, derived_quantity
            )
            assert isinstance(chi, Tensor)

        if chi.is_contiguous() is False:
            logger.debug(
                "Polarizability derivative: Re-enforcing contiguous memory "
                f"layout after autodiff (use_functorch={use_functorch})."
            )
            chi = chi.contiguous()

        return chi

    @cdec.numerical
    @cdec.requires_efield
    def pol_deriv_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
    ) -> Tensor:
        r"""
        Numerically calculate the cartesian polarizability derivative
        :math:`\chi`.

        .. math::

            \chi = \alpha' = \dfrac{\partial \alpha}{\partial R} = \dfrac{\partial^2 \mu}{\partial F \partial R} = \dfrac{\partial^3 E}{\partial^2 F \partial R}

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        step_size : float | int, optional
            Step size for the numerical derivative.

        Returns
        -------
        Tensor
            Polarizability derivative shape `(..., 3, 3, nat, 3)`.
        """
        # pylint: disable=import-outside-toplevel
        import gc

        # (..., 3, 3, nat, 3)
        deriv = torch.zeros(
            (*numbers.shape[:-1], 3, 3, *positions.shape[-2:]), **self.dd
        )
        logger.debug(
            "Polarizability derivative (numerical): Starting build " f"({deriv.shape})."
        )

        count = 1
        nsteps = 3 * numbers.shape[-1]
        for i in range(numbers.shape[-1]):
            for j in range(3):
                positions[..., i, j] += step_size
                r = self.polarizability_numerical(numbers, positions, chrg, spin)

                positions[..., i, j] -= 2 * step_size
                l = self.polarizability_numerical(numbers, positions, chrg, spin)

                positions[..., i, j] += step_size
                deriv[..., :, :, i, j] = 0.5 * (r - l) / step_size

                logger.debug(
                    "Polarizability numerical derivative: Step " f"{count}/{nsteps}"
                )
                count += 1

                gc.collect()
            gc.collect()

        logger.debug("Polarizability numerical derivative: All finished.")

        return deriv

    @cdec.requires_efield
    @cdec.requires_efield_grad
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
            Atomic numbers for all atoms in the system (shape: `(..., nat)`).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
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
            Hyper polarizability tensor of shape `(..., 3, 3, 3)`.
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

        # usually only after jacrev not contiguous
        if beta.is_contiguous() is False:
            logger.debug(
                "Hyperpolarizability: Re-enforcing contiguous memory "
                f"layout after autodiff (use_functorch={use_functorch})."
            )
            beta = beta.contiguous()

        return beta

    @cdec.numerical
    @cdec.requires_efield
    def hyperpol_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
    ) -> Tensor:
        r"""
        Numerically calculate the hyper polarizability tensor :math:`\beta`.

        .. math::

            \beta = \dfrac{\partial \alpha}{\partial F} = \dfrac{\partial^2 \mu}{\partial F^2} = \dfrac{\partial^3 E}{\partial^2 3}

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: `(..., nat)`).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        step_size : float | int, optional
            Step size for the numerical derivative.

        Returns
        -------
        Tensor
            Hyper polarizability tensor of shape `(..., 3, 3, 3)`.
        """
        # pylint: disable=import-outside-toplevel
        import gc

        # retrieve the efield interaction and the field and detach for gradient
        ef = self.interactions.get_interaction(efield.LABEL_EFIELD)
        _field = ef.field.clone()
        field = ef.field.detach().clone()
        self.interactions.update_efield(field=field)

        # (..., 3, 3, 3)
        deriv = torch.zeros(*(*numbers.shape[:-1], 3, 3, 3), **self.dd)
        logger.debug(
            f"Hyper Polarizability (numerical): Starting build ({deriv.shape})"
        )

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

            logger.debug(f"Hyper Polarizability (numerical): step {count}/{3}")
            count += 1

            gc.collect()

        # explicitly update field (to restore original field with possible grad)
        self.interactions.reset_efield()
        self.interactions.update_efield(field=_field)

        logger.debug("Hyper Polarizability (numerical): All finished.")

        return deriv

    # SPECTRA

    def ir(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
    ) -> vib.IRResult:
        """
        Calculate the frequencies and intensities of IR spectra.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: `(..., nat)`).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int | str | None
            Total charge. Defaults to `None`.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_functorch : bool, optional
            Whether to use functorch or the standard (slower) autograd.
            Defaults to `False`.

        Returns
        -------
        vib.IRResult
            Result container with frequencies (shape: `(..., nfreqs)`) and intensities (shape: `(..., nfreqs)`) of IR spectra.
        """
        OutputHandler.write_stdout("\nIR Spectrum")
        OutputHandler.write_stdout("-----------")
        logger.debug("IR spectrum: Start.")

        # run vibrational analysis first
        vib_res = self.vibration(numbers, positions, chrg, spin)

        # TODO: Figure out how to run func transforms 2x properly
        # (improve: Hessian does not need dipole integral but dipder does)
        self.integrals.reset_all()

        # calculate nuclear dipole derivative dmu/dR: (..., 3, nat, 3)
        dmu_dr = self.dipole_deriv(
            numbers, positions, chrg, spin, use_functorch=use_functorch
        )

        intensities = vib.ir_ints(dmu_dr, vib_res.modes)

        logger.debug("IR spectrum: All finished.")

        return vib.IRResult(vib_res.freqs, intensities)

    @cdec.numerical
    def ir_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
    ) -> vib.IRResult:
        """
        Numerically calculate the frequencies and intensities of IR spectra.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: `(..., nat)`).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int | str | None
            Total charge. Defaults to `None`.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        step_size : float | int, optional
            Step size for the numerical derivative.

        Returns
        -------
        vib.IRResult
            Result container with frequencies (shape: `(..., nfreqs)`) and intensities (shape: `(..., nfreqs)`) of IR spectra.
        """
        OutputHandler.write_stdout("\nIR Spectrum")
        OutputHandler.write_stdout("-----------")
        logger.debug("IR spectrum (numerical): Start.")

        # run vibrational analysis first
        freqs, modes = self.vibration_numerical(
            numbers, positions, chrg, spin, step_size=step_size
        )

        # calculate nuclear dipole derivative dmu/dR: (..., 3, nat, 3)
        dmu_dr = self.dipole_deriv_numerical(
            numbers, positions, chrg, spin, step_size=step_size
        )

        intensities = vib.ir_ints(dmu_dr, modes)

        logger.debug("IR spectrum (numerical): All finished.")

        return vib.IRResult(freqs, intensities)

    def raman(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
    ) -> vib.RamanResult:
        """
        Calculate the frequencies, static intensities and depolarization ratio
        of Raman spectra.
        Formula taken from `here <https://doi.org/10.1080/00268970701516412>`__.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: `(..., nat)`).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int | str | None
            Total charge. Defaults to `None`.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_functorch : bool, optional
            Whether to use functorch or the standard (slower) autograd.
            Defaults to `False`.

        Returns
        -------
        vib.RamanResult
            Result container with frequencies (shape: `(..., nfreqs)`),
            intensities (shape: `(..., nfreqs)`) and the depolarization ratio
            (shape: `(..., nfreqs)`) of Raman spectra.
        """
        OutputHandler.write_stdout("\nRaman Spectrum")
        OutputHandler.write_stdout("--------------")
        logger.debug("Raman spectrum: Start.")

        vib_res = self.vibration(
            numbers, positions, chrg, spin, use_functorch=use_functorch
        )

        # TODO: Figure out how to run func transforms 2x properly
        # (improve: Hessian does not need dipole integral but dipder does)
        self.integrals.reset_all()

        # d(..., 3, 3) / d(..., nat, 3) -> (..., 3, 3, nat, 3)
        da_dr = self.pol_deriv(
            numbers, positions, chrg, spin, use_functorch=use_functorch
        )

        intensities, depol = vib.raman_ints_depol(da_dr, vib_res.modes)

        logger.debug("Raman spectrum: All finished.")

        return vib.RamanResult(vib_res.freqs, intensities, depol)

    @cdec.numerical
    def raman_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
    ) -> vib.RamanResult:
        """
        Numerically calculate the frequencies, static intensities and
        depolarization ratio of Raman spectra.
        Formula taken from `here <https://doi.org/10.1080/00268970701516412>`__.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: `(..., nat)`).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int | str | None
            Total charge. Defaults to `None`.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        step_size : float | int, optional
            Step size for the numerical derivative.

        Returns
        -------
        vib.RamanResult
            Result container with frequencies (shape: `(..., nfreqs)`),
            intensities (shape: `(..., nfreqs)`) and the depolarization ratio
            (shape: `(..., nfreqs)`) of Raman spectra.
        """
        OutputHandler.write_stdout("\nRaman Spectrum")
        OutputHandler.write_stdout("--------------")
        logger.debug("Raman spectrum (numerical): All finished.")

        vib_res = self.vibration_numerical(
            numbers, positions, chrg, spin, step_size=step_size
        )

        # d(3, 3) / d(nat, 3) -> (3, 3, nat, 3) -> (3, 3, nat*3)
        da_dr = self.pol_deriv_numerical(
            numbers, positions, chrg, spin, step_size=step_size
        )

        intensities, depol = vib.raman_ints_depol(da_dr, vib_res.modes)

        logger.debug("Raman spectrum: All finished.")

        return vib.RamanResult(vib_res.freqs, intensities, depol)
