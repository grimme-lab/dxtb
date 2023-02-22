"""
Base calculator for the extended tight-binding model.
"""
from __future__ import annotations

import warnings

import torch

from .. import scf
from .._types import Any, Tensor, TensorLike
from ..basis import IndexHelper
from ..classical import Halogen, Repulsion, new_halogen, new_repulsion
from ..constants import defaults
from ..coulomb import new_es2, new_es3
from ..data import cov_rad_d3
from ..dispersion import Dispersion, new_dispersion
from ..integral import Overlap
from ..interaction import Interaction, InteractionList
from ..ncoord import exp_count, get_coordination_number
from ..param import Param, get_elem_angular
from ..utils import Timers, ToleranceWarning
from ..wavefunction import filling
from ..xtb.h0 import Hamiltonian
from .h0 import Hamiltonian


class Result(TensorLike):
    """
    Result container for singlepoint calculation.
    """

    charges: Tensor
    """Self-consistent orbital-resolved Mulliken partial charges."""

    coefficients: Tensor
    """LCAO-MO coefficients (eigenvectors of Fockian)."""

    density: Tensor
    """Density matrix."""

    dispersion: Tensor
    """Dispersion energy."""

    emo: Tensor
    """Energy of molecular orbitals (sorted by increasing energy)."""

    fenergy: Tensor
    """Atom-resolved electronic free energy from fractional occupation."""

    gradient: Tensor | None
    """Gradient of total energy w.r.t. positions"""

    halogen: Tensor
    """Halogen bond energy."""

    hamiltonian: Tensor
    """Full Hamiltonian matrix (H0 + H1)."""

    hcore: Tensor
    """Core Hamiltonian matrix (H0)."""

    occupation: Tensor
    """Orbital occupations."""

    overlap: Tensor
    """Overlap matrix."""

    potential: Tensor
    """Self-consistent orbital-resolved potential."""

    repulsion: Tensor
    """Repulsion energy."""

    scf: Tensor
    """Atom-resolved energy from the self-consistent field (SCF) calculation."""

    timer: Timers | None
    """Collection of timers for all steps."""

    total: Tensor
    """Total energy."""

    __slots__ = [
        "charges",
        "coefficients",
        "density",
        "dispersion",
        "emo",
        "fenergy",
        "gradient",
        "halogen",
        "hamiltonian",
        "hcore",
        "occupation",
        "overlap",
        "potential",
        "repulsion",
        "scf",
        "timer",
        "total",
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
        self.dispersion = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.repulsion = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.halogen = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.total = torch.zeros(shape, dtype=self.dtype, device=self.device)

    def __repr__(self) -> str:
        """Custom print representation showing all available slots."""
        return f"{self.__class__.__name__}({self.__slots__})"

    def print_energies(self, name: str = "Energy", width: int = 50) -> None:
        """Print energies in a table."""

        labels = {
            "dispersion": "Dispersion energy",
            "repulsion": "Repulsion energy",
            "halogen": "Halogen bond correction",
            "fenergy": "Electronic free energy",
            "scf": "Electronic Energy (SCF)",
        }

        print(f"{name:*^50}\n")
        print("{:<27}  {:<18}".format("Contribution", "Energy in a.u."))
        print(width * "-")

        tot = "Total Energy"
        total = torch.sum(self.total, dim=-1)

        for label, n in labels.items():
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

    interaction: Interaction
    """Interactions to minimize in self-consistent iterations."""

    ihelp: IndexHelper
    """Helper class for indexing."""

    opts: dict[str, Any]
    """Calculator options."""

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        interaction: Interaction | None = None,
        opts: dict[str, Any] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        dd = {"device": self.device, "dtype": self.dtype}

        # setup calculator options
        opts = opts if opts is not None else {}
        self.opts = {
            "fwd_options": {
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
                "verbosity": opts.get("verbosity", defaults.VERBOSITY),
            },
            "exclude": opts.get("exclude", defaults.EXCLUDE),
            "guess": opts.get("guess", defaults.GUESS),
            "spin": opts.get("spin", defaults.SPIN),
        }

        # set tolerances separately to catch unreasonably small values
        self.set_tol("f_tol", opts.get("xitorch_fatol", defaults.XITORCH_FATOL))
        self.set_tol("x_tol", opts.get("xitorch_xatol", defaults.XITORCH_XATOL))

        self.ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
        self.hamiltonian = Hamiltonian(numbers, par, self.ihelp, **dd)
        self.overlap = Overlap(numbers, par, self.ihelp, **dd)

        # setup self-consistent contributions
        es2 = new_es2(numbers, par, **dd) if "es2" not in self.opts["exclude"] else None
        es3 = new_es3(numbers, par, **dd) if "es3" not in self.opts["exclude"] else None
        self.interaction = InteractionList(es2, es3, interaction)

        # setup non-self-consistent contributions
        self.halogen = (
            new_halogen(numbers, par, **dd)
            if "hal" not in self.opts["exclude"]
            else None
        )
        self.dispersion = (
            new_dispersion(numbers, par, **dd)
            if "disp" not in self.opts["exclude"]
            else None
        )
        self.repulsion = (
            new_repulsion(numbers, par, **dd)
            if "rep" not in self.opts["exclude"]
            else None
        )

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

    def singlepoint(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor,
        timer: Timers | None = None,
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

        Returns
        -------
        Result
            Results.
        """

        result = Result(positions, device=self.device, dtype=self.dtype)
        timer = Timers() if timer is None else timer
        timer.start("singlepoint")

        if "scf" not in self.opts["exclude"]:
            # overlap
            timer.start("overlap")
            overlap = self.overlap.build(positions)
            result.overlap = overlap
            timer.stop("overlap")

            # Hamiltonian
            timer.start("h0")
            rcov = cov_rad_d3[numbers].to(self.device)
            cn = get_coordination_number(numbers, positions, exp_count, rcov)
            hcore = self.hamiltonian.build(positions, overlap, cn)
            result.hcore = hcore
            timer.stop("h0")

            # SCF
            timer.start("scf")

            # Obtain the reference occupations and total number of electrons
            n0 = self.hamiltonian.get_occupation()
            nel = torch.sum(n0, -1) - torch.sum(chrg, -1)

            # get alpha and beta electrons and occupation
            nab = filling.get_alpha_beta_occupation(nel, self.opts["spin"])
            occupation = filling.get_aufbau_occupation(
                hcore.new_tensor(hcore.shape[-1], dtype=torch.int64), nab
            )

            scf_results = scf.solve(
                numbers,
                positions,
                chrg,
                self.interaction,
                self.ihelp,
                self.opts["guess"],
                hcore,
                overlap,
                occupation,
                n0,
                fwd_options=self.opts["fwd_options"],
                scf_options=self.opts["scf_options"],
                use_potential=True,
            )
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
            timer.stop("scf")

        if self.halogen is not None:
            timer.start("halogen")
            cache_hal = self.halogen.get_cache(numbers, self.ihelp)
            result.halogen = self.halogen.get_energy(positions, cache_hal)
            result.total += result.halogen
            timer.stop("halogen")

        if self.dispersion is not None:
            timer.start("dispersion")
            cache_disp = self.dispersion.get_cache(numbers)
            result.dispersion = self.dispersion.get_energy(positions, cache_disp)
            result.total += result.dispersion
            timer.stop("dispersion")

        if self.repulsion is not None:
            timer.start("repulsion")
            cache_rep = self.repulsion.get_cache(numbers, self.ihelp)
            result.repulsion = self.repulsion.get_energy(positions, cache_rep)
            result.total += result.repulsion
            timer.stop("repulsion")

        timer.stop("singlepoint")
        result.timer = timer

        if self.opts["scf_options"]["verbosity"] > 0:
            result.print_energies()
        if self.opts["scf_options"]["verbosity"] > 1:
            print("")
            timer.print_times()

        return result
