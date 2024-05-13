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
Calculators: Energy
===================

Base calculator for the energy calculation of an extended tight-binding model.
"""

from __future__ import annotations

import logging

import torch
from tad_mctc.convert import any_to_tensor
from tad_mctc.io.checks import content_checks, shape_checks

from dxtb import OutputHandler
from dxtb import integrals as ints
from dxtb._src import scf
from dxtb._src.components.interactions.field import efield as efield
from dxtb._src.components.interactions.field import efieldgrad as efield_grad
from dxtb._src.constants import defaults
from dxtb._src.timing import timer
from dxtb._src.typing import Any, Tensor

from ..result import Result
from . import decorators as cdec
from .base import BaseCalculator

__all__ = ["EnergyCalculator"]


logger = logging.getLogger(__name__)


class EnergyCalculator(BaseCalculator):
    """
    Parametrized calculator defining the extended tight-binding model.

    This class provides the basic functionality for the extended tight-binding
    model. It provides methods for single point calculations, nuclear
    gradients, Hessians, molecular properties, and spectra.
    """

    implemented_properties: list[str] = ["energy"]
    """Names of implemented methods of the Calculator."""

    __slots__ = [
        "numbers",
        "cache",
        "opts",
        "classicals",
        "interactions",
        "integrals",
        "ihelp",
    ]

    def singlepoint(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
    ):
        """
        Entry point for performing single point calculations.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to 0.
        """
        # shape checks
        assert shape_checks(self.numbers, positions, allow_batched=True)
        assert content_checks(
            self.numbers, positions, self.opts.max_element, allow_batched=True
        )

        OutputHandler.write_stdout("Singlepoint ", v=3)

        chrg = any_to_tensor(chrg, **self.dd)
        if spin is not None:
            spin = any_to_tensor(spin, **self.dd)

        result = Result(positions, **self.dd)

        # CLASSICAL CONTRIBUTIONS

        if len(self.classicals.components) > 0:
            OutputHandler.write_stdout_nf(" - Classicals        ... ", v=3)
            timer.start("Classicals")

            ccaches = self.classicals.get_cache(self.numbers, self.ihelp)
            cenergies = self.classicals.get_energy(positions, ccaches)
            cenergy = torch.stack(list(cenergies.values())).sum(0)

            result.cenergies = cenergies
            result.total += cenergy

            timer.stop("Classicals")
            OutputHandler.write_stdout("done", v=3)

        if any(x in ["all", "scf"] for x in self.opts.exclude):
            return result

        # SELF-CONSISTENT FIELD PROCEDURE

        timer.start("Integrals")
        # overlap integral
        OutputHandler.write_stdout_nf(" - Overlap           ... ", v=3)
        timer.start("Overlap", parent_uid="Integrals")
        self.integrals.build_overlap(positions)
        timer.stop("Overlap")
        OutputHandler.write_stdout("done", v=3)

        # dipole integral
        if self.opts.ints.level >= ints.levels.INTLEVEL_DIPOLE:
            OutputHandler.write_stdout_nf(" - Dipole            ... ", v=3)
            timer.start("Dipole Integral", parent_uid="Integrals")
            self.integrals.build_dipole(positions)
            timer.stop("Dipole Integral")
            OutputHandler.write_stdout("done", v=3)

        # quadrupole integral
        if self.opts.ints.level >= ints.levels.INTLEVEL_QUADRUPOLE:
            OutputHandler.write_stdout_nf(" - Quadrupole        ... ", v=3)
            timer.start("Quadrupole Integral", parent_uid="Integrals")
            self.integrals.build_quadrupole(positions)
            timer.stop("Quadrupole Integral")
            OutputHandler.write_stdout("done", v=3)

        # Core Hamiltonian integral (requires overlap internally!)
        #
        # This should be the final integral, because the others are
        # potentially calculated on CPU (libcint) even in GPU runs.
        # To avoid unnecessary data transfer, the core Hamiltonian should
        # be last. Internally, the overlap integral is only transfered back
        # to GPU when all multipole integrals are calculated.
        OutputHandler.write_stdout_nf(" - Core Hamiltonian  ... ", v=3)
        timer.start("Core Hamiltonian", parent_uid="Integrals")
        self.integrals.build_hcore(positions)
        timer.stop("Core Hamiltonian")
        OutputHandler.write_stdout("done", v=3)

        timer.stop("Integrals")
        result.integrals = self.integrals
        # TODO: Flag for writing integrals to pt file or only return in results?

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
            numbers=self.numbers, positions=positions, ihelp=self.ihelp
        )
        timer.stop("Interaction Cache")
        OutputHandler.write_stdout("done", v=3)

        # SCF
        OutputHandler.write_stdout("\nStarting SCF Iterations...", v=3)

        scf_results = scf.solve(
            self.numbers,
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

        scf_energy = scf_results["energy"] + scf_results["fenergy"]
        total_energy = scf_energy + cenergy

        if self.opts.batch_mode == 0:
            OutputHandler.write_stdout(
                f"SCF Energy  : %.14f Hartree.",
                scf_results["energy"].sum(-1),
                v=2,
            )
            OutputHandler.write_stdout(
                f"Total Energy: %.14f Hartree.",
                total_energy.sum(-1),
                v=1,
            )

        self.cache["energy"] = total_energy
        self.cache["charges"] = scf_results["charges"]
        self.cache["iterations"] = scf_results["iterations"]

        self._ncalcs += 1
        return result

    @cdec.cache
    def energy(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
    ) -> Tensor:
        """
        Calculate the total energy :math:`E` of the system.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.

        Returns
        -------
        Tensor
            Total energy of the system (scalar value).
        """
        self.singlepoint(positions, chrg, spin)
        return self.cache["energy"].sum(-1)

    def calculate(
        self,
        properties: list[str],
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ):
        """
        Calculate the requested properties. This is more of a dispatcher method
        that calls the appropriate methods of the Calculator.

        Parameters
        ----------
        properties : list[str]
            List of properties to calculate.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.

        Returns
        -------
        dict
            Dictionary of calculated properties.
        """
        if "energy" in properties:
            self.energy(positions, chrg, spin, **kwargs)
