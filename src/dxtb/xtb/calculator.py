"""
Base calculator for the extended tight-binding model.
"""
from __future__ import annotations

import logging

import torch

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
from ..constants import defaults
from ..coulomb import new_es2, new_es3
from ..data import cov_rad_d3
from ..dispersion import Dispersion, new_dispersion
from ..interaction import Charges, Interaction, InteractionList, Potential
from ..interaction.external import field as efield
from ..param import Param, get_elem_angular
from ..utils import Timers, _jac
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
        classical: Sequence[Classical] = [],
        interaction: Sequence[Interaction] = [],
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
        if numbers.dtype not in (torch.long, torch.int16, torch.int32, torch.int64):
            raise ValueError(
                "Tensor for atomic numbers must be of integer of long type."
            )

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

        self.classicals = ClassicalList(
            halogen, dispersion, repulsion, *classical, timer=self.timer
        )

        #############
        # INTEGRALS #
        #############

        # figure out integral level from interactions
        self.intlevel = opts.get("int_level", defaults.INTLEVEL)

        # setup integral
        driver = self.opts["int_driver"]
        self.integrals = ints.Integrals(numbers, par, self.ihelp, driver=driver, **dd)

        if self.intlevel >= ints.INTLEVEL_OVERLAP:
            self.integrals.hcore = ints.Hamiltonian(numbers, par, self.ihelp, **dd)
            self.integrals.overlap = ints.Overlap(driver=driver, **dd)

        # TODO: This should get some extra validation by the config
        # (avoid PyTorch integral driver for multipole integrals)
        if self.intlevel >= ints.INTLEVEL_DIPOLE:
            self.integrals.dipole = ints.Dipole(driver=driver, **dd)

        if self.intlevel >= ints.INTLEVEL_QUADRUPOLE:
            self.integrals.quadrupole = ints.Quadrupole(driver=driver, **dd)

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
            logger.warn(
                f"Selected tolerance ({value:.2E}) is smaller than the "
                f"smallest value for the selected dtype ({self.dtype}, "
                f"{eps:.2E}). Switching to {eps:.2E} instead."
            )
            value = eps

        self.opts["fwd_options"][name] = value

    @property
    def intlevel(self) -> Tensor:
        if self.opts["int_level"] is None:
            raise ValueError("No overlap integral provided.")
        return self.opts["int_level"]

    @intlevel.setter
    def intlevel(self, value: int) -> None:
        if efield.LABEL_EFIELD in self.interactions.labels:
            value = max(2, value)

        self.opts["int_level"] = value

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
            Atomic numbers for all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
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
            # overlap integral
            self.timer.start("Overlap")
            self.integrals.build_overlap(positions)
            self.timer.stop("Overlap")

            # dipole integral
            if self.intlevel >= ints.INTLEVEL_DIPOLE:
                self.timer.start("Dipole Integral")
                self.integrals.build_dipole(positions)
                self.timer.stop("Dipole Integral")

            # quadrupole integral
            if self.intlevel >= ints.INTLEVEL_QUADRUPOLE:
                self.timer.start("Quadrupole Integral")
                self.integrals.build_quadrupole(positions)
                self.timer.stop("Quadrupole Integral")

            # TODO: Think about handling this case
            if self.integrals.hcore is None:
                raise RuntimeError
            if self.integrals.overlap is None:
                raise RuntimeError

            # Core Hamiltonian integral (requires overlap internally!)
            self.timer.start("h0", "Core Hamiltonian")
            rcov = cov_rad_d3[numbers].to(self.device)
            cn = ncoord.get_coordination_number(
                numbers, positions, ncoord.exp_count, rcov
            )
            hcore = self.integrals.build_hcore(positions, cn=cn)
            self.timer.stop("h0")

            # SCF
            self.timer.start("SCF")

            # Obtain the reference occupations and total number of electrons
            n0 = self.integrals.hcore.integral.get_occupation()
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
                self.integrals.matrices,
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
                result.overlap_grad = self.integrals.grad_overlap(positions)
                self.timer.stop("ograd")

                self.timer.start("hgrad", "Hamiltonian Gradient")
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

    def hessian(
        self, numbers: Tensor, positions: Tensor, chrg: Tensor, shape: str = "matrix"
    ) -> Tensor:
        return self.hessian_numerical(numbers, positions, chrg, shape)

    def hessian_numerical(
        self, numbers: Tensor, positions: Tensor, chrg: Tensor, shape: str = "matrix"
    ) -> Tensor:
        """
        Numerical Hessian

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        chrg : Tensor
            Total charge.
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
        tmp = self.opts["scf_options"]["verbosity"]
        self.opts["scf_options"]["verbosity"] = 0

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

        self.opts["scf_options"]["verbosity"] = tmp
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
        tmp = self.opts["scf_options"]["verbosity"]
        self.opts["scf_options"]["verbosity"] = 0

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

        print()
        print(ir_ints * 1378999.7790799031)

        logger.debug("IR spectrum: All finished.")

        self.opts["scf_options"]["verbosity"] = tmp
        return freqs * units.AU2RCM, ir_ints

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
