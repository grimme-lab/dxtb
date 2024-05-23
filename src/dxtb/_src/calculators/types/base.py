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
Calculators: Base Class
=======================

A base class for all calculators. All calculators should inherit from this
class and implement the :meth:`calculate` method and the corresponding methods
to calculate the properties specified within this :meth:`calculate`, as well as
the :attr:`implemented_properties` attribute.
"""
from __future__ import annotations

from abc import abstractmethod

import torch
from tad_mctc.exceptions import DtypeError

from dxtb import IndexHelper, OutputHandler
from dxtb import integrals as ints
from dxtb._src.calculators.properties.vibration import IRResult, RamanResult, VibResult
from dxtb._src.components.classicals import (
    Classical,
    ClassicalList,
    new_dispersion,
    new_halogen,
    new_repulsion,
)
from dxtb._src.components.interactions import Interaction, InteractionList
from dxtb._src.components.interactions.container import Charges, Potential
from dxtb._src.components.interactions.coulomb import new_es2, new_es3
from dxtb._src.components.interactions.field import efield as efield
from dxtb._src.components.interactions.field import efieldgrad as efield_grad
from dxtb._src.constants import defaults
from dxtb._src.param import Param
from dxtb._src.timing import timer
from dxtb._src.typing import Any, Self, Tensor, override
from dxtb.config import Config
from dxtb.integrals import Integrals
from dxtb.typing import Tensor, TensorLike

from .abc import GetPropertiesMixin, PropertyNotImplementedError


class CalculatorCache(TensorLike):
    """
    Cache for Calculator that extends TensorLike.

    This class provides caching functionality for storing multiple calculation results.
    """

    _cache_keys: dict[str, str | None]
    """Dictionary of cache keys and their corresponding hash values."""

    __slots__ = [
        "energy",
        #
        "forces",
        "hessian",
        "vibration",
        "normal_modes",
        "frequencies",
        #
        "dipole",
        "quadrupole",
        "polarizability",
        "hyperpolarizability",
        #
        "dipole_deriv",
        "pol_deriv",
        "ir",
        "ir_intensities",
        "raman",
        "raman_intensities",
        "raman_depol",
        #
        "hcore",
        "overlap",
        "dipint",
        "quadint",
        #
        "bond_orders",
        "charges",
        "coefficients",
        "density",
        "fock",
        "iterations",
        "mo_energies",
        "occupation",
        "potential",
        #
        "_cache_keys",
    ]

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        #
        energy: Tensor | None = None,
        forces: Tensor | None = None,
        hessian: Tensor | None = None,
        vibration: VibResult | None = None,
        normal_modes: Tensor | None = None,
        frequencies: Tensor | None = None,
        #
        dipole: Tensor | None = None,
        quadrupole: Tensor | None = None,
        polarizability: Tensor | None = None,
        hyperpolarizability: Tensor | None = None,
        #
        dipole_deriv: Tensor | None = None,
        pol_deriv: Tensor | None = None,
        ir: IRResult | None = None,
        ir_intensities: Tensor | None = None,
        raman: RamanResult | None = None,
        raman_intensities: Tensor | None = None,
        raman_depol: Tensor | None = None,
        #
        hcore: Tensor | None = None,
        overlap: ints.types.Overlap | None = None,
        dipint: ints.types.Dipole | None = None,
        quadint: ints.types.Quadrupole | None = None,
        #
        bond_orders: Tensor | None = None,
        coefficients: Tensor | None = None,
        charges: Charges | None = None,
        density: Tensor | None = None,
        fock: Tensor | None = None,
        mo_energies: Tensor | None = None,
        occupation: Tensor | None = None,
        potential: Potential | None = None,
        iterations: int | None = None,
        #
        _cache_keys: dict[str, str | None] | None = None,
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
        self.vibration = vibration
        self.normal_modes = normal_modes
        self.frequencies = frequencies

        self.dipole = dipole
        self.quadrupole = quadrupole
        self.polarizability = polarizability
        self.hyperpolarizability = hyperpolarizability

        self.dipole_deriv = dipole_deriv
        self.pol_deriv = pol_deriv
        self.ir = ir
        self.ir_intensities = ir_intensities
        self.raman = raman
        self.raman_intensities = raman_intensities
        self.raman_depol = raman_depol

        self.hcore = hcore
        self.overlap = overlap
        self.dipint = dipint
        self.quadint = quadint

        self.bond_orders = bond_orders
        self.coefficients = coefficients
        self.charges = charges
        self.density = density
        self.fock = fock
        self.iterations = iterations
        self.mo_energies = mo_energies
        self.occupation = occupation
        self.potential = potential

        self._cache_keys = (
            {prop: None for prop in self.__slots__ if prop != "_cache_keys"}
            if _cache_keys is None
            else _cache_keys
        )

    def __getitem__(self, key: str) -> Any:
        """
        Get an item from the cache.

        Parameters
        ----------
        key : str
            The key of the item to retrieve.

        Returns
        -------
        Tensor | None
            The value associated with the key, if it exists.
        """
        if key in self.__slots__:
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in Cache.")

    def __setitem__(self, key: str, value: Any) -> None:
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

        # also set the content of the spectroscopic results
        if key == "vibration":
            assert isinstance(value, VibResult)
            setattr(self, "frequencies", value.freqs)
            setattr(self, "normal_modes", value.modes)
        if key == "ir":
            assert isinstance(value, IRResult)
            setattr(self, "frequencies", value.freqs)
            setattr(self, "ir_intensities", value.ints)
        if key == "raman":
            assert isinstance(value, RamanResult)
            setattr(self, "frequencies", value.freqs)
            setattr(self, "raman_intensities", value.ints)
            setattr(self, "raman_depol", value.depol)

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

    def reset(self, key: str) -> None:
        """
        Clearing specific cached value by key.

        Parameters
        ----------
        key : str | None, optional
            The key to reset. If ``None``, all keys are reset. Defaults to
            ``None``.
        """
        setattr(self, key, None)

    def reset_all(self) -> None:
        """
        Reset the cache by clearing all cached values.

        Parameters
        ----------
        key : str | None, optional
            The key to reset. If ``None``, all keys are reset. Defaults to
            ``None``.
        """
        for key in self.__slots__:
            if key != "_cache_keys":
                setattr(self, key, None)

        self._cache_keys = {
            prop: None for prop in self.__slots__ if prop != "_cache_keys"
        }

    def list_cached_properties(self) -> list[str]:
        """
        List all cached properties.

        Returns
        -------
        list[str]
            List of cached properties.
        """
        return [
            key
            for key in self.__slots__
            if getattr(self, key) is not None and key != "_cache_keys"
        ]

    # cache validation

    def set_cache_key(self, key: str, hash: str) -> None:
        """
        Set the cache key for a specific property.

        Parameters
        ----------
        key : str
            The key of the item to set.
        hash : str
            The hash value to be associated with the key.
        """
        if self._cache_keys is None:
            raise RuntimeError("Cache keys have not been initialized.")

        if key not in self._cache_keys:
            raise KeyError(f"Key '{key}' cannot be set in Cache.")

        self._cache_keys[key] = hash

    def get_cache_key(self, key: str) -> str | None:
        """
        Get the cache key for a specific property.

        Parameters
        ----------
        key : str
            The key of the item to get.

        Returns
        -------
        str | None
            The hash value associated with the key.
        """
        if self._cache_keys is None:
            raise RuntimeError("Cache keys have not been initialized.")

        if key not in self._cache_keys:
            raise KeyError(f"Key '{key}' not found in '_cache_keys'.")

        return self._cache_keys[key]

    # printing

    def __str__(self) -> str:
        """Return a string representation of the Cache object."""
        return f"{self.__class__.__name__}({', '.join([f'{key}={getattr(self, key)!r}' for key in self.__slots__])})"

    def __repr__(self) -> str:
        """Return a representation of the Cache object."""
        return str(self)


class BaseCalculator(GetPropertiesMixin, TensorLike):
    """
    Base calculator for the extended tight-binding (xTB) models.
    """

    numbers: Tensor
    """Atomic numbers for all atoms in the system (shape: ``(..., nat)``)."""

    cache: CalculatorCache
    """Cache for storing multiple calculation results."""

    classicals: ClassicalList
    """Classical contributions."""

    interactions: InteractionList
    """Interactions to minimize in self-consistent iterations."""

    integrals: Integrals
    """Integrals for the extended tight-binding model."""

    ihelp: IndexHelper
    """Helper class for indexing."""

    opts: Config
    """Calculator configuration."""

    results: dict[str, Any]
    """Results container."""

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        *,
        classical: list[Classical] | tuple[Classical] | Classical | None = None,
        interaction: list[Interaction] | tuple[Interaction] | Interaction | None = None,
        opts: dict[str, Any] | Config | None = None,
        cache: CalculatorCache | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Instantiate the Calculator object with the following parameters:

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        par : Param
            Representation of an extended tight-binding model (full xtb
            parametrization). Decides energy contributions.
        classical : Sequence[Classical] | None, optional
            Additional classical contributions. Defaults to ``None``.
        interaction : Sequence[Interaction] | None, optional
            Additional self-consistent contributions (interactions).
            Defaults to ``None``.
        opts : dict[str, Any] | None, optional
            Calculator options. If ``None`` (default) is given, default options
            are used automatically.
        device : torch.device | None, optional
            Device to store the tensor on. If ``None`` (default), the default
            device is used.
        dtype : torch.dtype | None, optional
            Data type of the tensor. If ``None`` (default), the data type is
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

        allowed_dtypes = (torch.int16, torch.int32, torch.int64)
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
        self.cache = CalculatorCache(**dd) if cache is None else cache

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
            if not {"all", "es2"} & set(self.opts.exclude)
            else None
        )
        es3 = (
            new_es3(numbers, par, **dd)
            if not {"all", "es3"} & set(self.opts.exclude)
            else None
        )

        if interaction is None:
            self.interactions = InteractionList(es2, es3, **dd)
        elif isinstance(interaction, Interaction):
            self.interactions = InteractionList(es2, es3, interaction, **dd)
        elif isinstance(interaction, (list, tuple)):
            self.interactions = InteractionList(es2, es3, *interaction, **dd)
        else:
            raise TypeError(
                "Expected 'interaction' to be 'None' or of type 'Interaction', "
                "'list[Interaction]' or 'tuple[Interaction]', but got "
                f"'{type(interaction).__name__}'."
            )

        OutputHandler.write_stdout("done", v=4)

        ##############
        # CLASSICALS #
        ##############

        # setup non-self-consistent contributions
        OutputHandler.write_stdout_nf(" - Classicals        ... ", v=4)

        halogen = (
            new_halogen(numbers, par, **dd)
            if not {"all", "hal"} & set(self.opts.exclude)
            else None
        )
        dispersion = (
            new_dispersion(numbers, par, **dd)
            if not {"all", "disp"} & set(self.opts.exclude)
            else None
        )
        repulsion = (
            new_repulsion(numbers, par, **dd)
            if not {"all", "rep"} & set(self.opts.exclude)
            else None
        )

        if classical is None:
            self.classicals = ClassicalList(halogen, dispersion, repulsion, **dd)
        elif isinstance(classical, Classical):
            self.classicals = ClassicalList(
                halogen, dispersion, repulsion, classical, **dd
            )
        elif isinstance(classical, (list, tuple)):
            self.classicals = ClassicalList(
                halogen, dispersion, repulsion, *classical, **dd
            )
        else:
            raise TypeError(
                "Expected 'classical' to be 'None' or of type 'Classical', "
                "'list[Classical]' or 'tuple[Classical]', but got "
                f"'{type(classical).__name__}'."
            )

        OutputHandler.write_stdout("done", v=4)

        #############
        # INTEGRALS #
        #############

        OutputHandler.write_stdout_nf(" - Integrals         ... ", v=4)

        # figure out integral level from interactions
        if efield.LABEL_EFIELD in self.interactions.labels:
            if self.opts.ints.level < ints.levels.INTLEVEL_DIPOLE:
                OutputHandler.warn(
                    "Setting integral level to DIPOLE "
                    f"({ints.levels.INTLEVEL_DIPOLE}) due to electric field "
                    "interaction."
                )
            self.opts.ints.level = max(
                ints.levels.INTLEVEL_DIPOLE, self.opts.ints.level
            )
        if efield_grad.LABEL_EFIELD_GRAD in self.interactions.labels:
            if self.opts.ints.level < ints.levels.INTLEVEL_DIPOLE:
                OutputHandler.warn(
                    "Setting integral level to QUADRUPOLE "
                    f"{ints.levels.INTLEVEL_DIPOLE} due to electric field "
                    "gradient interaction."
                )
            self.opts.ints.level = max(
                ints.levels.INTLEVEL_QUADRUPOLE, self.opts.ints.level
            )

        # setup integral
        driver = self.opts.ints.driver
        self.integrals = ints.Integrals(
            numbers, par, self.ihelp, driver=driver, intlevel=self.opts.ints.level, **dd
        )

        if self.opts.ints.level >= ints.levels.INTLEVEL_OVERLAP:
            self.integrals.hcore = ints.types.HCore(numbers, par, self.ihelp, **dd)
            self.integrals.overlap = ints.types.Overlap(driver=driver, **dd)

        if self.opts.ints.level >= ints.levels.INTLEVEL_DIPOLE:
            self.integrals.dipole = ints.types.Dipole(driver=driver, **dd)

        if self.opts.ints.level >= ints.levels.INTLEVEL_QUADRUPOLE:
            self.integrals.quadrupole = ints.types.Quadrupole(driver=driver, **dd)

        OutputHandler.write_stdout("done\n", v=4)

        self._ncalcs = 0
        timer.stop("Calculator")

    def reset(self) -> None:
        """Reset the calculator to its initial state."""
        self.classicals.reset_all()
        self.interactions.reset_all()
        self.integrals.reset_all()
        self.cache.reset_all()

    @abstractmethod
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
        """

    def get_property(
        self,
        name: str,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        allow_calculation: bool = True,
        return_clone: bool = False,
        **kwargs: Any,
    ) -> Tensor | None:
        """
        Get the named property.

        Parameters
        ----------
        name : str
            Name of the property to get.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        allow_calculation : bool, optional
            If the property is not present, allow its calculation. This does
            not check if we even allow caching or if the inputs are the same.
            Use with caution. Defaults to ``True``.
        return_clone : bool, optional
            If True, return a clone of the property. Defaults to ``False``.
        """
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError(
                f"Property '{name}' not implemented. Use one of: "
                f"{self.implemented_properties}."
            )

        # If we do not allow calculation and do not have the property in the
        # cache, there's nothing we can do. Note that this does not check if we
        # even allow caching or if the inputs are the same.
        if allow_calculation is False and name not in self.cache:
            return None

        # All the cache checks are handled deep within `calculate`. No need to
        # do it here as well.
        self.calculate([name], positions, chrg=chrg, spin=spin, **kwargs)

        # For some reason the calculator was not able to do what we want...
        if name not in self.cache:
            raise PropertyNotImplementedError(
                f"Property '{name}' not present after calculation. "
                "This seems like an internal error. (Maybe the method you "
                "are calling has no cache decorator?)"
            )

        if return_clone is False:
            return self.cache[name]

        result = self.cache[name]
        if isinstance(result, Tensor):
            result = result.clone()
        return result

    @override
    def type(self, dtype: torch.dtype) -> Self:
        """
        Returns a copy of the class instance with specified floating point type.

        This method overrides the usual approach because the
        :class:`dxtb.Calculator`'s arguments and slots differ significantly.
        Hence, it is not practical to instantiate a new copy.

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
            If the ``__slots__`` attribute is not set in the class.
        DtypeError
            If the specified dtype is not allowed.
        """

        if self.dtype == dtype:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `type` method requires setting ``__slots__`` in the "
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

        This method overrides the usual approach because the
        :class:`dxtb.Calculator`'s arguments and slots differ significantly.
        Hence, it is not practical to instantiate a new copy.

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
            If the ``__slots__`` attribute is not set in the class.
        """
        if self.device == device:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `to` method requires setting ``__slots__`` in the "
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
