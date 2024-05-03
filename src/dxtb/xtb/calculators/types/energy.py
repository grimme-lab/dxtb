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
from tad_mctc.exceptions import DtypeError
from tad_mctc.io.checks import content_checks, shape_checks

from dxtb import integral as ints
from dxtb import scf
from dxtb.basis import IndexHelper
from dxtb.components.classicals import (
    Classical,
    ClassicalList,
    new_dispersion,
    new_halogen,
    new_repulsion,
)
from dxtb.components.interactions import Interaction, InteractionList
from dxtb.components.interactions.coulomb import new_es2, new_es3
from dxtb.components.interactions.field import efield as efield
from dxtb.components.interactions.field import efieldgrad as efield_grad
from dxtb.config import Config
from dxtb.constants import defaults
from dxtb.io import OutputHandler
from dxtb.param import Param
from dxtb.timing import timer
from dxtb.typing import Any, Self, Sequence, Tensor, TensorLike, override

from ..result import Result
from . import decorators as cdec

__all__ = ["EnergyCalculator"]


logger = logging.getLogger(__name__)


class EnergyCalculator(TensorLike):
    """
    Parametrized calculator defining the extended tight-binding model.

    This class provides the basic functionality for the extended tight-binding
    model. It provides methods for single point calculations, nuclear
    gradients, Hessians, molecular properties, and spectra.
    """

    numbers: Tensor
    """Atomic numbers for all atoms in the system (shape: `(..., nat)`)."""

    cache: Cache
    """Cache for storing multiple calculation results."""

    interactions: InteractionList
    """Interactions to minimize in self-consistent iterations."""

    classicals: ClassicalList
    """Classical contributions."""

    integrals: ints.Integrals
    """Integrals for the extended tight-binding model."""

    ihelp: IndexHelper
    """Helper class for indexing."""

    opts: Config
    """Calculator configuration."""

    __slots__ = [
        "numbers",
        "cache",
        "opts",
        "classicals",
        "interactions",
        "integrals",
    ]

    class Cache(TensorLike):
        """
        Cache for Calculator that extends TensorLike.

        This class provides caching functionality for storing multiple calculation results.
        """

        __slots__ = [
            "energy",
            "forces",
            "hessian",
            "dipole",
            "quadrupole",
            "polarizability",
            "hyperpolarizability",
        ]

        def __init__(
            self,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
            energy: Tensor | None = None,
            forces: Tensor | None = None,
            hessian: Tensor | None = None,
            dipole: Tensor | None = None,
            quadrupole: Tensor | None = None,
            polarizability: Tensor | None = None,
            hyperpolarizability: Tensor | None = None,
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
            self.energy = energy
            self.forces = forces
            self.hessian = hessian
            self.dipole = dipole
            self.quadrupole = quadrupole
            self.polarizability = polarizability
            self.hyperpolarizability = hyperpolarizability

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
        classical: Sequence[Classical] | None = None,
        interaction: Sequence[Interaction] | None = None,
        opts: dict[str, Any] | Config | None = None,
        cache: Cache | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
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
            opts = dict(opts)
            OutputHandler.verbosity = opts.pop("verbosity", 1)

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
        self.numbers = numbers

        super().__init__(device, dtype)
        dd = {"device": self.device, "dtype": self.dtype}

        # setup calculator options
        if isinstance(opts, dict):
            opts = Config(**opts, **dd)
        self.opts = opts

        # create cache
        self.cache = self.Cache(**dd) if cache is None else cache

        if self.opts.batch_mode == 0 and numbers.ndim > 1:
            self.opts.batch_mode = 1

        self.ihelp = IndexHelper.from_numbers(numbers, par, self.opts.batch_mode)

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

        self.interactions = InteractionList(es2, es3, *(interaction or ()), **dd)

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

        self.classicals = ClassicalList(
            halogen, dispersion, repulsion, *(classical or ()), **dd
        )

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
        self.integrals = ints.Integrals(
            numbers, par, self.ihelp, driver=driver, intlevel=self.opts.ints.level, **dd
        )

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
        """Reset the calculator to its initial state."""
        self.classicals.reset_all()
        self.interactions.reset_all()
        self.integrals.reset_all()

    def singlepoint(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
    ) -> Result:
        """
        Entry point for performing single point calculations.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to 0.

        Returns
        -------
        Result
            Results container.
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
            result.cenergies = cenergies
            result.total += torch.stack(list(cenergies.values())).sum(0)

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

        # TIMERS AND PRINTOUT
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
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.

        Returns
        -------
        Tensor
            Total energy of the system (scalar value).
        """
        return self.singlepoint(positions, chrg, spin).total.sum(-1)

    @override
    def type(self, dtype: torch.dtype) -> Self:
        """
        Returns a copy of the class instance with specified floating point type.

        This method overrides the usual approach because the `Calculator`s
        arguments and slots differ significantly. Hence, it is not practical to
        instantiate a new copy.

        Parameters
        ----------
        dtype : torch.dtype
            Floating point type.

        Returns
        -------
        Self
            A copy of the class instance with the specified dtype.

        Raises
        ------
        RuntimeError
            If the `__slots__` attribute is not set in the class.
        DtypeError
            If the specified dtype is not allowed.
        """

        if self.dtype == dtype:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `type` method requires setting `__slots__` in the "
                f"'{self.__class__.__name__}' class."
            )

        if dtype not in self.allowed_dtypes:
            raise DtypeError(
                f"Only '{self.allowed_dtypes}' allowed (received '{dtype}')."
            )

        self.classicals = self.classicals.type(dtype)
        self.interactions = self.interactions.type(dtype)
        self.integrals = self.integrals.type(dtype)
        self.cache = self.cache.type(dtype)

        # simple override in config
        self.opts.dtype = dtype

        # hard override of the dtype in TensorLike
        self.override_dtype(dtype)

        return self

    @override
    def to(self, device: torch.device) -> Self:
        """
        Returns a copy of the class instance on the specified device.

        This method overrides the usual approach because the `Calculator`s
        arguments and slots differ significantly. Hence, it is not practical to
        instantiate a new copy.

        Parameters
        ----------
        device : torch.device
            Device to store the tensor on.

        Returns
        -------
        Self
            A copy of the class instance on the specified device.

        Raises
        ------
        RuntimeError
            If the `__slots__` attribute is not set in the class.
        """
        if self.device == device:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `to` method requires setting `__slots__` in the "
                f"'{self.__class__.__name__}' class."
            )

        self.classicals = self.classicals.to(device)
        self.interactions = self.interactions.to(device)
        self.integrals = self.integrals.to(device)
        self.cache = self.cache.to(device)

        # simple override in config
        self.opts.device = device

        # hard override of the dtype in TensorLike
        self.override_device(device)

        return self
