# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
Spin Polarization
===============================

Implementation of spin-interactions for spin-polarized tigh-binding.
The spin-components are represented as the difference between
alpha and beta population, resulting in a magnetization density.
"""

from __future__ import annotations

import torch
from tad_mctc.math import einsum

from dxtb import IndexHelper
from dxtb._src.typing import Any, Slicers, Tensor, override

from ..base import Interaction, InteractionCache

__all__ = ["SpinPolarisation", "LABEL_SPINPOLARISATION"]


LABEL_SPINPOLARISATION = "SpinPolarisation"
"""Label for the :class:`.SpinPolarisation` interaction, coinciding with the class name."""


class SpinPolarisationCache(InteractionCache):
    """
    Restart data for the :class:`.SpinPolarisation` interaction.

    Note
    ----
    The spin constants are given in the class constructor.
    """

    __store: Store | None
    """Storage for cache (required for culling)."""

    wll: Tensor
    """Matrix of spin"""  # TO DO

    __slots__ = ["__store", "wll", "shell_resolved"]

    def __init__(
        self,
        wll: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=device if device is None else wll.device,
            dtype=dtype if dtype is None else wll.dtype,
        )
        self.wll = wll
        self.__store = None

    class Store:
        """
        Storage container for cache containing ``__slots__`` before culling.
        """

        wll: Tensor
        """"""  # TO DO

        def __init__(
            self,
            wll: Tensor,
        ) -> None:
            self.wll = wll

    def cull(
        self,
        conv: Tensor,
        slicers: Slicers,
    ) -> None:
        if self.__store is None:
            self.__store = self.Store(self.wll)

        slicer = slicers["shell"]
        self.wll = self.wll[[~conv, *slicer]]

    def restore(self):
        if self.__store is None:
            raise RuntimeError("Nothing to restore. Store is empty.")

        self.wll = self.__store.wll


class SpinPolarisation(Interaction):
    """
    Spin Polarisation (:class: `.SpinPolarisation`).

    This interaction operates on the magnetization channel of the
    spin-resolved charges, computing the spin-polarisation energy and
    potential.  The base-class routing (via ``spin_channel = 1``) ensures
    that the magnetization charges are extracted before being passed to
    the ``get_monopole_shell_*`` methods, and that the returned potential
    is placed into the correct spin channel of the combined potential
    tensor.
    """

    spinconst: Tensor

    spin_channel: int | None = 1
    """Read magnetization charges (channel 1) when nspin > 1."""

    __slots__ = [
        "spinconst",
    ]

    def __init__(
        self,
        spinconst: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.spinconst = spinconst

    @override
    def get_cache(
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
    ) -> SpinPolarisationCache:
        """
        Create restart data for individual interactions

        Parameters
        -----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        --------
        SpinPolarisationCache
            Restart data for the interaction.

        """
        if numbers is None:
            raise ValueError("Atomic numbers are required for spinpol cache.")

        if ihelp is None:
            raise ValueError(
                "IndexHelper is required for spinpol cache creation."
            )

        cachvars = numbers.detach().clone()

        if self.cache_is_latest(cachvars) is True:
            if not isinstance(self.cache, SpinPolarisationCache):
                raise TypeError(
                    f"Cache in {self.label} is not of type '{self.label}."
                    "Cache'. This can only happen if you manually manipulate "
                    "the cache."
                )
            return self.cache

        # if the cache is built, store the cachevar for validation
        self._cachevars = cachvars

        lidx = torch.tensor(
            [[0, 1, 3], [1, 2, 4], [3, 4, 5]], device=self.device
        )

        wll = torch.zeros((ihelp.nsh, ihelp.nsh), **self.dd)

        for atom_idx, number in enumerate(numbers):
            for ishell in range(ihelp.shells_per_atom[atom_idx]):
                for jshell in range(ihelp.shells_per_atom[atom_idx]):
                    ishell_idx = ihelp.shell_index[atom_idx] + ishell
                    jshell_idx = ihelp.shell_index[atom_idx] + jshell
                    l1 = ihelp.angular[jshell_idx]
                    l2 = ihelp.angular[ishell_idx]
                    wll[jshell_idx, ishell_idx] = self.spinconst[
                        atom_idx, lidx[l1, l2]
                    ]

        self.cache = SpinPolarisationCache(wll)
        return self.cache

    @override
    def get_monopole_shell_energy(
        self, cache: SpinPolarisationCache, qsh: Tensor, **_: Any
    ) -> Tensor:
        """
        Calculate the spin-polarisation energy from magnetization charges.

        The base class extracts the magnetization channel before calling
        this method, so ``qsh`` is single-channel ``(..., nsh)``.

        Parameters
        ----------
        cache : SpinPolarisationCache
            Restart data for the interaction.
            Contains the spin-constant matrix ``wll``.
        qsh : Tensor
            Shell-resolved magnetization charges (shape: ``(..., nsh)``).

        Returns
        -------
        Tensor
            Shell-wise spin-polarisation energy (shape: ``(..., nsh)``).
        """
        return 0.5 * qsh * einsum("...k,...ik->...i", qsh, cache.wll)

    @override
    def get_monopole_shell_potential(
        self, cache: SpinPolarisationCache, qsh: Tensor, **_: Any
    ) -> Tensor:
        """
        Calculate the spin-polarisation potential from magnetization charges.

        The base class extracts the magnetization channel before calling
        this method, so ``qsh`` is single-channel ``(..., nsh)``.

        Parameters
        ----------
        cache : SpinPolarisationCache
            Restart data for the interaction.
            Contains the spin-constant matrix ``wll``.
        qsh : Tensor
            Shell-resolved magnetization charges (shape: ``(..., nsh)``).

        Returns
        -------
        Tensor
            Shell-wise spin-polarisation potential (shape: ``(..., nsh)``).
        """
        return einsum("...k,...ik->...i", qsh, cache.wll)

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(spinconst={self.spinconst})"

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)
