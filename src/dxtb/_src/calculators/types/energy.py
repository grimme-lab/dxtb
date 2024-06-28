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
from dxtb._src.constants import defaults
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.timing import timer
from dxtb._src.typing import Any, Tensor
from dxtb._src.utils.tensors import tensor_id

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

    # all from an SCF
    implemented_properties: list[str] = [
        "bond_orders",
        "energy",
        "coefficients",
        "charges",
        "density",
        "iterations",
        "mo_energies",
        "occupation",
        "potential",
    ]
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
        **kwargs: Any,
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

        # get the hashed key for the cache from all arguments
        hashed_key = ""
        all_args = (positions, chrg, spin) + tuple(kwargs.values())
        for i, arg in enumerate(all_args):
            sep = "_" if i > 0 else ""
            if isinstance(arg, Tensor):
                hashed_key += f"{sep}{tensor_id(arg)}"
            else:
                hashed_key += f"{sep}{arg}"

        chrg = any_to_tensor(chrg, **self.dd)
        if spin is not None:
            spin = any_to_tensor(spin, **self.dd)

        result = Result(positions, **self.dd)

        ###########################
        # CLASSICAL CONTRIBUTIONS #
        ###########################

        if len(self.classicals.components) > 0:
            OutputHandler.write_stdout_nf(" - Classicals        ... ", v=3)
            timer.start("Classicals")

            ccaches = self.classicals.get_cache(self.numbers, self.ihelp)
            cenergies = self.classicals.get_energy(positions, ccaches)
            result.cenergies = cenergies
            result.total += torch.stack(list(cenergies.values())).sum(0)

            timer.stop("Classicals")
            OutputHandler.write_stdout("done", v=3)

        if {"all", "scf"} & set(self.opts.exclude):
            self.cache["energy"] = result.total

            return result

        #############
        # INTEGRALS #
        #############

        timer.start("Integrals")

        intmats = IntegralMatrices(**self.dd)

        # overlap integral (always required, even without Hückel Hamiltonian)
        OutputHandler.write_stdout_nf(" - Overlap           ... ", v=3)
        timer.start("Overlap", parent_uid="Integrals")
        intmats.overlap = self.integrals.build_overlap(positions)
        timer.stop("Overlap")
        OutputHandler.write_stdout("done", v=3)

        if self.integrals.overlap is None:
            raise RuntimeError("Overlap setup failed. SCF cannot be run.")
        if self.integrals.overlap.matrix is None:
            raise RuntimeError("Overlap calculation failed. SCF cannot be run.")

        write_overlap = kwargs.get("write_overlap", False)
        if write_overlap is not False:
            assert self.integrals.overlap is not None
            self.integrals.overlap.to_pt(write_overlap)

        # dipole integral
        if self.opts.ints.level >= ints.levels.INTLEVEL_DIPOLE:
            OutputHandler.write_stdout_nf(" - Dipole            ... ", v=3)
            timer.start("Dipole Integral", parent_uid="Integrals")
            intmats.dipole = self.integrals.build_dipole(positions)
            timer.stop("Dipole Integral")
            OutputHandler.write_stdout("done", v=3)

            write_dipole = kwargs.get("write_dipole", False)
            if write_dipole is not False:
                assert self.integrals.dipole is not None
                self.integrals.dipole.to_pt(write_dipole)

        # quadrupole integral
        if self.opts.ints.level >= ints.levels.INTLEVEL_QUADRUPOLE:
            OutputHandler.write_stdout_nf(" - Quadrupole        ... ", v=3)
            timer.start("Quadrupole Integral", parent_uid="Integrals")
            intmats.quadrupole = self.integrals.build_quadrupole(positions)
            timer.stop("Quadrupole Integral")
            OutputHandler.write_stdout("done", v=3)

            write_quad = kwargs.get("write_quadrupole", False)
            if write_quad is not False:
                assert self.integrals.quadrupole is not None
                self.integrals.quadrupole.to_pt(write_quad)

        # Core Hamiltonian integral (requires overlap internally!)
        #
        # This should be the final integral, because the others are
        # potentially calculated on CPU (libcint) even in GPU runs.
        # To avoid unnecessary data transfer, the core Hamiltonian should
        # be last. Internally, the overlap integral is only transfered back
        # to GPU when all multipole integrals are calculated.
        if self.opts.ints.level >= ints.levels.INTLEVEL_HCORE:
            OutputHandler.write_stdout_nf(" - Core Hamiltonian  ... ", v=3)
            timer.start("Core Hamiltonian", parent_uid="Integrals")
            intmats.hcore = self.integrals.build_hcore(positions)
            timer.stop("Core Hamiltonian")
            OutputHandler.write_stdout("done", v=3)

            write_hcore = kwargs.get("write_hcore", False)
            if write_hcore is not False:
                assert self.integrals.hcore is not None
                self.integrals.hcore.to_pt(write_hcore)

        # While one can theoretically skip the core Hamiltonian, the
        # current implementation does not account for this case because the
        # reference occupation is necessary for the SCF procedure.
        if self.integrals.hcore is None or self.integrals.hcore.matrix is None:
            raise NotImplementedError(
                "Core Hamiltonian missing. Skipping the Core Hamiltonian in "
                "the SCF is currently not supported. Please increase the "
                "integral level to at least '2'. Currently, the level is set "
                f"to '{self.opts.ints.level}'."
            )

        # finalize integrals
        timer.stop("Integrals")
        intmats = intmats.to(self.device)
        result.integrals = intmats

        ###################################
        # SELF-CONSISTENT FIELD PROCEDURE #
        ###################################

        old_cuda_sync = timer.cuda_sync
        timer.cuda_sync = kwargs.get(
            "cuda_sync_in_scf", False if self.device.type == "cpu" else True
        )
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
            intmats,
            self.integrals.hcore.integral.refocc,
        )

        timer.stop("SCF")
        timer.cuda_sync = old_cuda_sync
        OutputHandler.write_stdout(
            f"SCF finished in {scf_results['iterations']} iterations.", v=3
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
        result.scf = scf_results["energy"]
        result.fenergy = scf_results["fenergy"]

        scf_energy = scf_results["energy"] + scf_results["fenergy"]
        result.total += scf_energy

        if self.opts.batch_mode == 0:
            OutputHandler.write_stdout(
                f"SCF Energy  : %.14f Hartree.",
                scf_results["energy"].sum(-1),
                v=2,
            )
            OutputHandler.write_stdout(
                f"Total Energy: %.14f Hartree.",
                result.total.sum(-1),
                v=1,
            )

        # Store results. Energy always stored.
        self.cache["energy"] = result.total

        copts = self.opts.cache.store

        if kwargs.get("store_charges", copts.charges):
            self.cache["charges"] = scf_results["charges"]
            self.cache.set_cache_key("charges", "charges:" + hashed_key)
        if kwargs.get("store_coefficients", copts.coefficients):
            self.cache["coefficients"] = scf_results["coefficients"]
            self.cache.set_cache_key("coefficients", "coefficients:" + hashed_key)
        if kwargs.get("store_density", copts.density):
            self.cache["density"] = scf_results["density"]
            self.cache.set_cache_key("density", "density:" + hashed_key)
        if kwargs.get("store_iterations", copts.iterations):
            self.cache["iterations"] = scf_results["iterations"]
            self.cache.set_cache_key("iterations", "iterations:" + hashed_key)
        if kwargs.get("store_mo_energies", copts.mo_energies):
            self.cache["mo_energies"] = scf_results["emo"]
            self.cache.set_cache_key("mo_energies", "mo_energies:" + hashed_key)
        if kwargs.get("store_occupation", copts.occupation):
            self.cache["occupation"] = scf_results["occupation"]
            self.cache.set_cache_key("occupation", "occupation:" + hashed_key)
        if kwargs.get("store_potential", copts.potential):
            self.cache["potential"] = scf_results["potential"]
            self.cache.set_cache_key("potential", "potential:" + hashed_key)

        if kwargs.get("store_fock", copts.fock):
            self.cache["fock"] = scf_results["hamiltonian"]
        if kwargs.get("store_hcore", copts.hcore):
            self.cache["hcore"] = self.integrals.hcore
        if kwargs.get("store_overlap", copts.overlap):
            self.cache["overlap"] = self.integrals.overlap
        if kwargs.get("store_dipole", copts.dipole):
            self.cache["dipint"] = self.integrals.dipole
        if kwargs.get("store_quadrupole", copts.quadrupole):
            self.cache["quadint"] = self.integrals.quadrupole

        self._ncalcs += 1
        return result

    @cdec.cache
    def energy(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
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
        self.singlepoint(positions, chrg, spin, **kwargs)
        e = self.cache["energy"]

        if e is None:
            raise RuntimeError(
                "Energy not found in cache after singlepoint calculation. "
                "This should not happen; the `singlepoint` method should "
                "always write at least the energy to the cache (even "
                "without caching enabled). Please report this issue."
            )

        return e.sum(-1)

    @cdec.cache
    def bond_orders(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        """
        Calculate the (Wiberg) bond order matrix.

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
            Bond order matrix.
        """
        self.singlepoint(positions, chrg, spin, **kwargs)

        overlap = self.cache["overlap"]
        if overlap is None:
            raise RuntimeError(
                "Overlap matrix not found in cache. The overlap is not saved "
                "per default. Enable saving either via the calculator options "
                "(`calc.opts.cache.store.overlap = True`) or by passing the "
                "`store_overlap=True` keyword argument to called method, e.g., "
                "`calc.energy(positions, store_overlap=True)"
            )

        assert isinstance(overlap, ints.types.Overlap)
        assert overlap.matrix is not None

        density = self.cache["density"]
        if density is None:
            raise RuntimeError(
                "Density matrix not found in cache. The density is not saved "
                "per default. Enable saving either via the calculator options "
                "(`calc.opts.cache.store.density = True`) or by passing the "
                "`store_density=True` keyword argument to called method, e.g., "
                "`calc.energy(positions, store_density=True)"
            )

        # pylint: disable=import-outside-toplevel
        from dxtb._src.wavefunction.wiberg import get_bond_order

        return get_bond_order(overlap.matrix, density, self.ihelp)

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
        if self.opts.cache.enabled is False:
            self.cache.reset_all()

        # treat bond orders separately for better error message
        if "bond_orders" in properties:
            self.bond_orders(positions, chrg, spin, **kwargs)

        props = self.get_implemented_properties()
        props.remove("bond_orders")
        if set(props) & set(properties):
            self.energy(positions, chrg, spin, **kwargs)
