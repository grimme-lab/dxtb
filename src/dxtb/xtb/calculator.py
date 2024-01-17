"""
Base calculator for the extended tight-binding model.
"""
from __future__ import annotations

import logging

import torch
from tad_mctc.convert import any_to_tensor

from .. import integral as ints
from .. import ncoord, properties, scf
from .._types import Any, Sequence, Tensor, TensorLike
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
from ..io import OutputHandler
from ..param import Param, get_elem_angular
from ..timing import timer
from ..utils import _jac
from ..wavefunction import filling

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
                f" indexing: '{', '.join([str(x) for x in allowed_dtypes])}', "
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
            self.intlevel = max(2, self.intlevel)

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
        if len(self.classicals.classicals) > 0:
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
                if len(self.interactions.interactions) > 0:
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

    def forces_numerical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
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
        step = 1.0e-5
        for i in range(numbers.shape[-1]):
            for j in range(3):
                pos[..., i, j] += step
                gr = self.energy(numbers, pos, chrg, spin)

                pos[..., i, j] -= 2 * step
                gl = self.energy(numbers, pos, chrg, spin)

                pos[..., i, j] += step
                jac[..., i, j] = 0.5 * (gr - gl) / step

                logger.debug(f"Numerical Forces: step {count}/{nsteps}")
                count += 1

        logger.debug("Numerical Forces: All finished.")

        OutputHandler.verbosity = tmp
        return -jac

    def forces(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int | None = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
    ) -> Tensor:
        if positions.requires_grad is False:
            raise RuntimeError("Position tensor needs `requires_grad=True`.")

        logger.debug(f"Autodiff Forces: Starting Calculation.")

        # pylint: disable=import-outside-toplevel
        from tad_mctc.autograd import jacrev

        # jacrev requires a scalar from `self.energy`!
        jac_func = jacrev(self.energy, argnums=1)
        jac = jac_func(numbers, positions, chrg, spin)
        assert isinstance(jac, Tensor)

        logger.debug("Autodiff Forces: All finished.")
        return -jac

    def hessian(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int | None = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        shape: str = "matrix",
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
            Number of unpaired electrons. Defaults to 0.
        shape : str, optional
            Output shape of Hessian. Defaults to "matrix".

        Returns
        -------
        Tensor
            Hessian matrix.

        Raises
        ------
        RuntimeError
            Positions tensor does not have `requires_grad=True`.
        """
        if positions.requires_grad is False:
            raise RuntimeError("Position tensor needs `requires_grad=True`.")

        logger.debug(f"Autodiff Hessian: Starting Calculation.")

        # pylint: disable=import-outside-toplevel
        from tad_mctc.autograd import jacrev

        # jacrev requires a scalar from `self.energy`!
        hess_func = jacrev(jacrev(self.energy, argnums=1), argnums=1)
        hess = hess_func(numbers, positions, chrg, spin)
        assert isinstance(hess, Tensor)

        # reshape (nb, nat, 3, nat, 3) to (nb, nat*3, nat*3)
        if shape == "matrix":
            s = [*numbers.shape[:-1], *2 * [3 * numbers.shape[-1]]]
            hess = hess.reshape(*s)

        logger.debug("Autodiff Hessian: All finished.")
        return hess

    def hessian2(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int | None = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        shape: str = "matrix",
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
        if shape == "matrix":
            s = [*numbers.shape[:-1], *2 * [3 * numbers.shape[-1]]]
            hess = hess.reshape(*s)

        logger.debug("Autodiff Hessian: All finished.")
        return hess

    def hessian_numerical(
        self, numbers: Tensor, positions: Tensor, chrg: Tensor, shape: str = "matrix"
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
            Total charge. Defaults to `None`
        shape : str, optional
            Output shape of Hessian. Defaults to "matrix".

        Returns
        -------
        Tensor
            Hessian.
        """
        # pylint: disable=import-outside-toplevel
        import gc

        def _gradfcn(pos: Tensor) -> Tensor:
            pos.requires_grad_(True)
            result = self.singlepoint(numbers, pos, chrg, grad=True)
            pos.detach_()
            return result.total_grad.detach()

        # imortant: detach for gradient
        pos = positions.detach().clone()

        # turn off printing in numerical hessian
        tmp = OutputHandler.verbosity
        OutputHandler.verbosity = 0

        hess = torch.zeros(
            *(*positions.shape, *positions.shape[-2:]),
            **{"device": positions.device, "dtype": positions.dtype},
        )

        logger.debug(f"Numerical Hessian: Starting build (size {hess.shape})")

        count = 1
        nsteps = 3 * numbers.shape[-1]
        step = 1.0e-5
        for i in range(numbers.shape[-1]):
            for j in range(3):
                pos[..., i, j] += step
                gr = _gradfcn(pos)

                pos[..., i, j] -= 2 * step
                gl = _gradfcn(pos)

                pos[..., i, j] += step
                hess[..., :, :, i, j] = 0.5 * (gr - gl) / step

                logger.debug(f"Numerical Hessian: step {count}/{nsteps}")
                count += 1

            gc.collect()
        gc.collect()

        # reshape (nb, nat, 3, nat, 3) to (nb, nat*3, nat*3)
        if shape == "matrix":
            s = [*numbers.shape[:-1], *2 * [3 * numbers.shape[-1]]]
            hess = hess.reshape(*s)

        logger.debug("Numerical Hessian: All finished.")

        OutputHandler.verbosity = tmp
        return hess

    def vibration(
        self, numbers: Tensor, positions: Tensor, chrg: Tensor, project: bool = True
    ) -> tuple[Tensor, Tensor]:
        hess = self.hessian_numerical(numbers, positions, chrg, shape="matrix")
        return properties.frequencies(numbers, positions, hess, project=project)

    def dipole(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor,
        use_autograd: bool = False,
    ) -> Tensor:
        # check if electric field is given in interactions
        if efield.LABEL_EFIELD not in self.interactions.labels:
            raise RuntimeError(
                "Dipole moment requires an electric field. Add the "
                f"'{efield.LABEL_EFIELD}' interaction to the Calculator."
            )

        # run single point and check if integral is populated
        result = self.singlepoint(numbers, positions, chrg)
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

        if use_autograd is False:
            # dip = properties.dipole(
            # numbers, positions, result.density, self.integrals.dipole
            # )
            qat = self.ihelp.reduce_orbital_to_atom(result.charges.mono)
            dip = properties.moments.dipole(
                qat, positions, result.density, dipint.matrix
            )
        else:
            # retrieve the efield interaction and the field
            ef = self.interactions.get_interaction(efield.LABEL_EFIELD)
            field = ef.field

            if field.requires_grad is False:
                raise RuntimeError("Field tensor needs `requires_grad=True`.")

            # calculate electric dipole contribution from xtb energy: -de/dE
            energy = result.total.sum(-1)
            dip = -_jac(energy, field)

        return dip

    def quadrupole(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor,
        use_autograd: bool = False,
    ) -> Tensor:
        if use_autograd is False:
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

        # check if electric field is given in interactions
        if efield.LABEL_EFIELD not in self.interactions.labels:
            raise RuntimeError(
                "Quadrupole moment requires an electric field. Add the "
                f"'{efield.LABEL_EFIELD}' interaction to the Calculator."
            )

        if positions.requires_grad is False:
            raise RuntimeError("Position tensor needs `requires_grad=True`.")

        # retrieve the efield interaction and the field
        ef = self.interactions.get_interaction(efield.LABEL_EFIELD)
        field = ef.field
        field_grad = ef.field_grad

        if field.requires_grad is False:
            raise RuntimeError("Field vector needs `requires_grad=True`.")
        if field_grad is None:
            raise RuntimeError("Field gradient must be set.")
        if field_grad.requires_grad is False:
            raise RuntimeError("Field gradient needs `requires_grad=True`.")

        energy = self.singlepoint(numbers, positions, chrg).total.sum(-1)

        print("")
        e_quad = _jac(energy, field_grad)
        print("quad_moment\n", e_quad)

        e_quad = e_quad.view(3, 3)
        print("")
        print("")

        print("quad_moment", e_quad.shape)

        cart = torch.empty((6), device=self.device, dtype=self.dtype)

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

    def polarizability(
        self, numbers: Tensor, positions: Tensor, chrg: Tensor
    ) -> Tensor:
        # check if electric field is given in interactions
        if efield.LABEL_EFIELD not in self.interactions.labels:
            raise RuntimeError(
                "Polarizability moment requires an electric field. Add the "
                f"'{efield.LABEL_EFIELD}' interaction to the Calculator."
            )

        # retrieve the efield interaction and the field
        ef = self.interactions.get_interaction(efield.LABEL_EFIELD)
        field = ef.field

        if field.requires_grad is False:
            raise RuntimeError("Field vector needs `requires_grad=True`.")

        # above checks are also run in dipole but only if via autograd
        mu = self.dipole(numbers, positions, chrg)

        # 3x3 polarizability tensor
        alpha = _jac(mu, field)

        return alpha

    def ir_spectrum(
        self, numbers: Tensor, positions: Tensor, chrg: Tensor
    ) -> tuple[Tensor, Tensor]:
        if positions.requires_grad is False:
            raise RuntimeError("Position tensor needs `requires_grad=True`.")

        # dipole moment with gradient tracking
        with torch.enable_grad():
            mu = self.dipole(numbers, positions, chrg)

        # calculate vibrational frequencies and normal modes
        freqs, modes = self.vibration(numbers, positions, chrg, project=True)

        return properties.ir(mu, positions, freqs, modes)

    def ir_spectrum_num(self, numbers: Tensor, positions: Tensor, chrg: Tensor):
        logger.debug("IR spectrum: Start.")

        # run vibrational analysis first
        hess = self.hessian_numerical(numbers, positions, chrg, shape="matrix")
        freqs, modes = properties.frequencies(numbers, positions, hess)

        # pylint: disable=import-outside-toplevel
        import gc

        # important: use new/separate position tensor
        pos = positions.detach().clone()

        # turn off printing in numerical hessian
        tmp = OutputHandler.verbosity
        OutputHandler.verbosity = 0

        dmu_dr = torch.zeros(
            (3, *positions.shape[-2:]),
            **{"device": positions.device, "dtype": positions.dtype},
        )

        logger.debug(
            "IR spectrum: Start building numerical dipole derivative: "
            f"(size {dmu_dr.shape})."
        )

        count = 1
        nsteps = 3 * numbers.shape[-1]
        step = 1.0e-5
        for i in range(positions.shape[-2]):
            for j in range(3):
                pos[i, j] += step
                er = self.dipole(numbers, pos, chrg)

                pos[i, j] -= 2 * step
                el = self.dipole(numbers, pos, chrg)

                pos[i, j] += step
                dmu_dr[:, i, j] = 0.5 * (er - el) / step

                logger.debug(
                    "IR spectrum: Numerical dipole derivative step " f"{count}/{nsteps}"
                )
                count += 1

            gc.collect()
        gc.collect()

        logger.debug("IR spectrum: Numerical dipole derivative finished")

        dmu_dr = dmu_dr.view(3, -1)
        dmu_dq = torch.matmul(dmu_dr, modes)  # (ndim, nfreqs)
        ir_ints = torch.einsum("...df,...df->...f", dmu_dq, dmu_dq)  # (nfreqs,)

        # TODO: Figure out unit
        from ..constants import units

        # print()
        # print(ir_ints * 1378999.7790799031)

        logger.debug("IR spectrum: All finished.")

        OutputHandler.verbosity = tmp
        return freqs * units.AU2RCM, ir_ints * 1378999.7790799031

    def raman_spectrum(
        self, numbers: Tensor, positions: Tensor, chrg: Tensor
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

        # check if electric field is given in interactions
        if efield.LABEL_EFIELD not in self.interactions.labels:
            raise RuntimeError(
                "Raman spectrum requires an electric field. Add the "
                f"'{efield.LABEL_EFIELD}' interaction to the Calculator."
            )

        # retrieve the efield interaction and the field
        ef = self.interactions.get_interaction(efield.LABEL_EFIELD)
        field = ef.field

        if field.requires_grad is False:
            raise RuntimeError("Field tensor needs `requires_grad=True`.")

        alpha = self.polarizability(numbers, positions, chrg)
        freqs, modes = self.vibration(numbers, positions, chrg, project=True)
        return properties.raman(alpha, freqs, modes)
