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
Calculators: Analytical
=======================

Calculator for the extended tight-binding model with analytical gradients.
"""

from __future__ import annotations

import torch
from tad_mctc.convert import any_to_tensor

from dxtb import OutputHandler
from dxtb._src import ncoord, scf
from dxtb._src.components.interactions.field import efield as efield
from dxtb._src.constants import defaults
from dxtb._src.timing import timer
from dxtb._src.typing import Any, Tensor
from dxtb.integrals import levels as intlvl

from ..result import Result
from . import decorators as cdec
from .energy import EnergyCalculator

__all__ = ["AnalyticalCalculator"]


class AnalyticalCalculator(EnergyCalculator):
    """
    Parametrized calculator defining the extended tight-binding model.

    This class provides analytical formulas and/or gradients for certain
    properties.
    """

    implemented_properties = EnergyCalculator.implemented_properties + [
        "forces",
        "dipole",
    ]
    """Names of implemented methods of the Calculator."""

    @cdec.requires_positions_grad
    @cdec.cache
    def forces_analytical(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        r"""
        Calculate the nuclear forces :math:`f` via AD.

        .. math::

            f = -\dfrac{\partial E}{\partial R}

        One can calculate the Jacobian either row-by-row using the standard
        :func:`torch.autograd.grad` with unit vectors in the VJP or using
        :mod:`torch.func`'s function transforms (e.g.,
        :func:`torch.func.jacrev`).

        Note
        ----
        Using :mod:`torch.func`'s function transforms can apparently be only
        used once. Hence, for example, the Hessian and the dipole derivatives
        cannot be both calculated with functorch.

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
            Atomic forces of shape ``(..., nat, 3)``.
        """
        OutputHandler.write_stdout("Singlepoint ", v=3)

        chrg = any_to_tensor(chrg, **self.dd)
        if spin is not None:
            spin = any_to_tensor(spin, **self.dd)

        total_grad = torch.zeros(positions.shape, **self.dd)
        result = Result(positions, **self.dd)

        # CLASSICAL CONTRIBUTIONS

        if len(self.classicals.components) > 0:
            OutputHandler.write_stdout_nf(" - Classicals        ... ", v=3)
            timer.start("Classicals")

            ccaches = self.classicals.get_cache(self.numbers, self.ihelp)
            cenergies = self.classicals.get_energy(positions, ccaches)
            result.cenergies = cenergies
            result.total += torch.stack(list(cenergies.values())).sum(0)

            timer.stop("Classicals")
            OutputHandler.write_stdout("done", v=3)
            OutputHandler.write_stdout_nf(" - Classicals Grad    ... ", v=3)
            timer.start("Classicals Gradient")

            cgradients = self.classicals.get_gradient(cenergies, positions)
            total_grad += torch.stack(list(cgradients.values())).sum(0)

            timer.stop("Classicals Gradient")

        if any(x in ["all", "scf"] for x in self.opts.exclude):
            return -total_grad

        # SELF-CONSISTENT FIELD PROCEDURE

        timer.start("Integrals")
        # overlap integral
        OutputHandler.write_stdout_nf(" - Overlap           ... ", v=3)
        timer.start("Overlap", parent_uid="Integrals")
        self.integrals.build_overlap(positions)
        timer.stop("Overlap")
        OutputHandler.write_stdout("done", v=3)

        # dipole integral
        if self.opts.ints.level >= intlvl.INTLEVEL_DIPOLE:
            OutputHandler.write_stdout_nf(" - Dipole            ... ", v=3)
            timer.start("Dipole Integral", parent_uid="Integrals")
            self.integrals.build_dipole(positions)
            timer.stop("Dipole Integral")
            OutputHandler.write_stdout("done", v=3)

        # quadrupole integral
        if self.opts.ints.level >= intlvl.INTLEVEL_QUADRUPOLE:
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
        result.scf += scf_results["energy"]
        result.total += scf_results["energy"] + scf_results["fenergy"]
        result.iter = scf_results["iterations"]

        if self.opts.batch_mode == 0:
            OutputHandler.write_stdout(
                f"SCF Energy  : {result.scf.sum(-1):.14f} Hartree.",
                v=2,
            )
            OutputHandler.write_stdout(
                f"Total Energy: {result.total.sum(-1):.14f} Hartree.", v=1
            )

        if len(self.interactions.components) > 0:
            timer.start("igrad", "Interaction Gradient")

            # charges should be detached
            interaction_grad = self.interactions.get_gradient(
                result.charges, positions, icaches, self.ihelp
            )
            total_grad += interaction_grad
            timer.stop("igrad")

        timer.start("ograd", "Overlap Gradient")
        overlap_grad = self.integrals.grad_overlap(positions)
        timer.stop("ograd")

        timer.start("hgrad", "Hamiltonian Gradient")
        wmat = scf.get_density(
            result.coefficients,
            result.occupation.sum(-2),
            emo=result.emo,
        )

        cn = ncoord.cn_d3(self.numbers, positions)
        dedcn, dedr = self.integrals.hcore.integral.get_gradient(
            positions,
            self.integrals.matrices.overlap,
            overlap_grad,
            result.density,
            wmat,
            result.potential,
            cn,
        )

        # CN gradient
        dcndr = ncoord.cn_d3_gradient(self.numbers, positions)
        dcn = ncoord.get_dcn(dcndr, dedcn)

        # sum up hamiltonian gradient and CN gradient
        hamiltonian_grad = dedr + dcn
        total_grad += hamiltonian_grad
        timer.stop("hgrad")

        # DEVNOTE: The cache decorator only writes the quantity specified by
        # the function name in the cache. For this, the quantity must be
        # returned from the function. All other quantities must be entered
        # explicitly into the cache.
        self.cache["energy"] = result.total
        self.cache["charges"] = result.charges
        self.cache["iterations"] = result.iter

        self._ncalcs += 1

        return -total_grad

    @cdec.requires_efield
    @cdec.cache
    def dipole_analytical(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        *_,  # absorb stuff
        **kwargs: Any,
    ) -> Tensor:
        r"""
        Analytically calculate the electric dipole moment :math:`\mu`.

        .. math::

            \mu = \dfrac{\partial E}{\partial F}

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
            Electric dipole moment of shape `(..., 3)`.
        """
        # run single point and check if integral is populated
        result = self.singlepoint(positions, chrg, spin)

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

        # pylint: disable=import-outside-toplevel
        from ..properties.moments.dip import dipole

        # dip = properties.dipole(
        # numbers, positions, result.density, self.integrals.dipole
        # )
        qat = self.ihelp.reduce_orbital_to_atom(result.charges.mono)
        dip = dipole(qat, positions, result.density, dipint.matrix)
        return dip

    def calculate(
        self,
        properties: list[str],
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs,
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
        """
        if "forces" in properties:
            kwargs.pop("grad_mode")
            self.forces_analytical(positions, chrg, spin, **kwargs)

        if "dipole" in properties:
            self.dipole_analytical(positions, chrg, spin, **kwargs)
