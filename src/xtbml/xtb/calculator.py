"""
Base calculator for the extended tight-binding model.
"""

import torch

from .. import scf
from ..basis import IndexHelper
from ..classical import Halogen, Repulsion, new_halogen, new_repulsion
from ..constants import defaults
from ..coulomb import new_es2, new_es3
from ..data import cov_rad_d3
from ..dispersion import Dispersion, new_dispersion
from ..interaction import Interaction, InteractionList
from ..ncoord import exp_count, get_coordination_number
from ..param import Param, get_elem_angular
from ..typing import Any, Tensor
from ..utils import Timers
from ..wavefunction import filling
from ..xtb.h0 import Hamiltonian
from .h0 import Hamiltonian


class Result:
    """
    Result container for singlepoint calculation.
    """

    charges: Tensor
    """Self-consistent orbital-resolved Mulliken partial charges"""

    density: Tensor
    """Density matrix."""

    dispersion: Tensor
    """Dispersion energy."""

    emo: Tensor
    """Energy of molecular orbitals (sorted by increasing energy)."""

    fenergy: Tensor
    """Atom-resolved electronic free energy from fractional occupation."""

    halogen: Tensor
    """Halogen bond energy."""

    hamiltonian: Tensor
    """Full Hamiltonian matrix (H0 + H1)."""

    hcore: Tensor
    """Core Hamiltonian matrix (H0)."""

    overlap: Tensor
    """Overlap matrix."""

    repulsion: Tensor
    """Repulsion energy."""

    scf: Tensor
    """Atom-resolved energy from the self-consistent field (SCF) calculation."""

    total: Tensor
    """Total energy."""

    __slots__ = [
        "charges",
        "density",
        "dispersion",
        "emo",
        "fenergy",
        "halogen",
        "hamiltonian",
        "hcore",
        "overlap",
        "repulsion",
        "scf",
        "total",
    ]

    def __init__(self, positions: Tensor):
        shape = positions.shape[:-1]
        device = positions.device
        dtype = positions.dtype

        self.scf = torch.zeros(shape, dtype=dtype, device=device)
        self.fenergy = torch.zeros(shape, dtype=dtype, device=device)
        self.dispersion = torch.zeros(shape, dtype=dtype, device=device)
        self.repulsion = torch.zeros(shape, dtype=dtype, device=device)
        self.halogen = torch.zeros(shape, dtype=dtype, device=device)
        self.total = torch.zeros(shape, dtype=dtype, device=device)


class Calculator:
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

    def __init__(
        self,
        numbers: Tensor,
        positions: Tensor,
        par: Param,
        interaction: Interaction | None = None,
    ) -> None:
        self.ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
        self.hamiltonian = Hamiltonian(numbers, positions, par, self.ihelp)

        es2 = new_es2(numbers, positions, par)
        es3 = new_es3(numbers, positions, par)
        self.interaction = InteractionList(es2, es3, interaction)

        self.halogen = new_halogen(numbers, positions, par)
        self.dispersion = new_dispersion(numbers, positions, par)
        self.repulsion = new_repulsion(numbers, positions, par)

    def singlepoint(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor,
        opts: dict[str, Any],
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
        opts : dict[str, Any]
            Options.

        Returns
        -------
        Result
            Results.
        """

        result = Result(positions)
        timer = Timers()
        timer.start("total")

        # overlap
        timer.start("overlap")
        overlap = self.hamiltonian.overlap()
        result.overlap = overlap
        timer.stop("overlap")

        # Hamiltonian
        timer.start("h0")
        rcov = cov_rad_d3[numbers]
        cn = get_coordination_number(numbers, positions, exp_count, rcov)
        hcore = self.hamiltonian.build(overlap, cn)
        result.hcore = hcore
        timer.stop("h0")

        # SCF
        timer.start("scf")

        # Obtain the reference occupations and total number of electrons
        n0 = self.hamiltonian.get_occupation()
        nel = torch.sum(n0, -1) - torch.sum(chrg, -1)
        occupation = 2 * filling.get_aufbau_occupation(
            hcore.new_tensor(hcore.shape[-1], dtype=torch.int64),
            nel / 2,
        )

        fwd_options = {
            "verbose": opts.get("verbosity", defaults.VERBOSITY),
            "maxiter": opts.get("maxiter", defaults.MAXITER),
        }
        scf_options = {
            "etemp": opts.get("etemp", defaults.ETEMP),
            "fermi_maxiter": opts.get("fermi_maxiter", defaults.FERMI_MAXITER),
            "fermi_thresh": opts.get("fermi_thresh", defaults.THRESH),
            "fermi_fenergy_partition": opts.get(
                "fermi_fenergy_partition", defaults.FERMI_FENERGY_PARTITION
            ),
        }
        guess = opts.get("guess", defaults.GUESS)

        scf_results = scf.solve(
            numbers,
            positions,
            chrg,
            self.interaction,
            self.ihelp,
            guess,
            hcore,
            overlap,
            occupation,
            n0,
            fwd_options=fwd_options,
            scf_options=scf_options,
            use_potential=True,
        )
        result.charges = scf_results["charges"]
        result.density = scf_results["density"]
        result.emo = scf_results["emo"]
        result.hamiltonian = scf_results["hamiltonian"]
        result.scf += scf_results["energy"]
        result.fenergy = scf_results["fenergy"]
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
            # cache_disp = self.dispersion.get_cache(numbers, self.ihelp)
            # result.dispersion = self.dispersion.get_energy(positions, cache_disp)
            result.dispersion = self.dispersion.get_energy(positions)
            result.total += result.dispersion
            timer.stop("dispersion")

        if self.repulsion is not None:
            timer.start("repulsion")
            cache_rep = self.repulsion.get_cache(numbers, self.ihelp)
            result.repulsion = self.repulsion.get_energy(positions, cache_rep)
            result.total += result.repulsion
            timer.stop("repulsion")

        timer.stop("total")

        if fwd_options["verbose"] > 0:
            timer.print_times()

        return result
