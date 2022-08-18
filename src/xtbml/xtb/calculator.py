# This file is part of xtbml.

"""
Base calculator for the extended tight-binding model.
"""

from __future__ import annotations
import torch


from .. import scf
from ..basis import IndexHelper
from ..classical.halogen import Halogen, new_halogen
from ..classical.repulsion import Repulsion, new_repulsion
from ..coulomb import secondorder, thirdorder, averaging_function
from ..dispersion import new_dispersion, Dispersion
from ..data import cov_rad_d3
from ..interaction import Interaction, InteractionList
from ..ncoord import ncoord
from ..param import Param, get_elem_param, get_elem_angular
from ..typing import Tensor
from ..wavefunction import filling
from ..xtb.h0 import Hamiltonian
from ..utils import Timers


class Result:
    """
    Result container for singlepoint calculation.
    """

    scf: Tensor
    """Energy from the self-consistent field (SCF) calculation."""

    dispersion: Tensor
    """Dispersion energy."""

    repulsion: Tensor
    """Repulsion energy."""

    halogen: Tensor
    """Halogen bond energy."""

    total: Tensor
    """Total energy."""

    hcore: Tensor
    """Core Hamiltonian matrix (H0)."""

    hamiltonian: Tensor
    """Full Hamiltonian matrix (H0 + H1)."""

    overlap: Tensor
    """Overlap matrix."""

    density: Tensor
    """Density matrix."""

    __slots__ = [
        "scf",
        "dispersion",
        "repulsion",
        "halogen",
        "total",
        "hcore",
        "hamiltonian",
        "overlap",
        "density",
    ]

    def __init__(self, positions: Tensor):
        shape = positions.shape[:-1]
        device = positions.device
        dtype = positions.dtype

        self.scf = torch.zeros(shape, dtype=dtype, device=device)
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

        if par.charge is None:
            raise ValueError("No charge parameters provided.")

        es2 = secondorder.ES2(
            hubbard=get_elem_param(
                torch.unique(numbers),
                par.element,
                "gam",
                device=positions.device,
                dtype=positions.dtype,
            ),
            lhubbard=get_elem_param(
                torch.unique(numbers),
                par.element,
                "lgam",
                device=positions.device,
                dtype=positions.dtype,
            ),
            average=averaging_function[par.charge.effective.average],
            gexp=torch.tensor(par.charge.effective.gexp),
        )
        es3 = thirdorder.ES3(
            hubbard_derivs=get_elem_param(
                torch.unique(numbers),
                par.element,
                "gam3",
                device=positions.device,
                dtype=positions.dtype,
            ),
        )

        self.interaction = InteractionList(es2, es3, interaction)

        self.halogen = new_halogen(numbers, positions, par)
        self.dispersion = new_dispersion(numbers, positions, par)
        self.repulsion = new_repulsion(numbers, positions, par)

    def singlepoint(
        self,
        numbers: Tensor,
        positions: Tensor,
        charges: Tensor,
        verbosity: int = 1,
    ) -> Result:
        """
        Entry point for performing single point calculations.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers.
        positions : Tensor
            Atomic positions.
        ihelp : IndexHelper
            Index mapping for the basis set.

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
        cn = ncoord.get_coordination_number(numbers, positions, ncoord.exp_count, rcov)
        hcore = self.hamiltonian.build(overlap, cn)
        result.hcore = hcore
        timer.stop("h0")

        # SCF
        timer.start("scf")

        # Obtain the reference occupations and total number of electrons
        n0 = self.hamiltonian.get_occupation()
        nel = torch.sum(n0, -1) - torch.sum(charges, -1)
        occupation = 2 * filling.get_aufbau_occupation(
            hcore.new_tensor(hcore.shape[-1], dtype=torch.int64),
            nel / 2,
        )

        fwd_options = {
            "verbose": verbosity > 0,
            "maxiter": 20,
        }
        scf_results = scf.solve(
            numbers,
            positions,
            self.interaction,
            self.ihelp,
            hcore,
            overlap,
            occupation,
            n0,
            fwd_options=fwd_options,
            use_potential=True,
        )
        result.scf += scf_results["energy"]
        result.total += scf_results["energy"]
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
            result.dispersion = self.dispersion.get_energy()
            result.total += result.dispersion
            timer.stop("dispersion")

        if self.repulsion is not None:
            timer.start("repulsion")
            cache_rep = self.repulsion.get_cache(numbers, self.ihelp)
            result.repulsion = self.repulsion.get_energy(positions, cache_rep)
            result.total += result.repulsion
            timer.stop("repulsion")

        timer.stop("total")

        if verbosity > -1:
            timer.print_times()

        return result
