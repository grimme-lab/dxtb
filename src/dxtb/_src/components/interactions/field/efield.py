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
External Fields: Electric Field
===============================

Interaction of the charge density with an external electric field.
"""

from __future__ import annotations

import torch
from tad_mctc.math import einsum

from dxtb._src.typing import Any, Slicers, Tensor, override
from dxtb._src.typing.exceptions import DeviceError, DtypeError

from ..base import Interaction, InteractionCache
from ..container import Charges

__all__ = ["ElectricField", "LABEL_EFIELD", "new_efield"]


LABEL_EFIELD = "ElectricField"
"""Label for the 'ElectricField' interaction, coinciding with the class name."""


class ElectricFieldCache(InteractionCache):
    """
    Restart data for the electric field interaction.
    """

    __store: Store | None
    """Storage for cache (required for culling)."""

    vat: Tensor
    """
    Atom-resolved monopolar potental from instantaneous electric field.
    """

    vdp: Tensor
    """
    Atom-resolved dipolar potential from instantaneous electric field.
    """

    __slots__ = ["__store", "vat", "vdp"]

    def __init__(
        self,
        vat: Tensor,
        vdp: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=device if device is None else vat.device,
            dtype=dtype if dtype is None else vat.dtype,
        )
        self.vat = vat
        self.vdp = vdp
        self.__store = None

    class Store:
        """
        Storage container for cache containing ``__slots__`` before culling.
        """

        vat: Tensor
        """
        Atom-resolved monopolar potental from instantaneous electric field.
        """

        vdp: Tensor
        """
        Atom-resolved dipolar potential from instantaneous electric field.
        """

        def __init__(self, vat: Tensor, vdp: Tensor) -> None:
            self.vat = vat
            self.vdp = vdp

    def cull(self, conv: Tensor, slicers: Slicers) -> None:
        if self.__store is None:
            self.__store = self.Store(self.vat, self.vdp)

        slicer = slicers["atom"]
        self.vat = self.vat[[~conv, *slicer]]
        self.vdp = self.vdp[[~conv, *slicer, ...]]

    def restore(self) -> None:
        if self.__store is None:
            raise RuntimeError("Nothing to restore. Store is empty.")

        self.vat = self.__store.vat
        self.vdp = self.__store.vdp


class ElectricField(Interaction):
    """
    Instantaneous electric field.
    """

    field: Tensor
    """Instantaneous electric field vector."""

    __slots__ = ["field"]

    def __init__(
        self,
        field: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            device=device if device is None else field.device,
            dtype=dtype if dtype is None else field.dtype,
        )
        self.field = field

    @override
    def get_cache(self, positions: Tensor, **_: Any) -> ElectricFieldCache:
        """
        Create restart data for individual interactions.

        Returns
        -------
        ElectricFieldCache
            Restart data for the interaction.

        Note
        ----
        If this interaction is evaluated within the `InteractionList`, `numbers`
        and `IndexHelper` will be passed as argument, too. The `**_` in the
        argument list will absorb those unnecessary arguments which are given
        as keyword-only arguments (see `Interaction.get_cache()`).
        """
        cachvars = (positions.detach().clone(), self.field.detach().clone())

        if self.cache_is_latest(cachvars) is True:
            if not isinstance(self.cache, ElectricFieldCache):
                raise TypeError(
                    f"Cache in {self.label} is not of type '{self.label}."
                    "Cache'. This can only happen if you manually manipulate "
                    "the cache."
                )
            return self.cache

        self._cachevars = cachvars

        # (nbatch, natoms, 3) * (3) -> (nbatch, natoms)
        vat = einsum("...ik,k->...i", positions, self.field)

        # (nbatch, natoms, 3)
        vdp = self.field.expand_as(positions)

        self.cache = ElectricFieldCache(vat, vdp)
        return self.cache

    @override
    def get_atom_energy(self, charges: Tensor, cache: ElectricFieldCache) -> Tensor:
        """
        Calculate the monopolar contribution of the electric field energy.

        Parameters
        ----------
        charges : Tensor
            Atomic charges of all atoms.
        cache : ElectricFieldCache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-wise electric field interaction energies.
        """
        return -cache.vat * charges

    @override
    def get_dipole_energy(self, charges: Tensor, cache: ElectricFieldCache) -> Tensor:
        """
        Calculate the dipolar contribution of the electric field energy.

        Parameters
        ----------
        charges : Tensor
            Atomic dipole moments of all atoms.
        cache : ElectricFieldCache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-wise electric field interaction energies.
        """

        # equivalent: torch.sum(-cache.vdp * charges, dim=-1)
        return einsum("...ix,...ix->...i", -cache.vdp, charges)

    @override
    def get_atom_potential(self, _: Charges, cache: ElectricFieldCache) -> Tensor:
        """
        Calculate the electric field potential.

        Parameters
        ----------
        charges : Tensor
            Atomic charges of all atoms (not required).
        cache : ElectricFieldCache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-wise electric field potential.
        """
        return -cache.vat

    @override
    def get_dipole_potential(self, _: Charges, cache: ElectricFieldCache) -> Tensor:
        """
        Calculate the electric field dipole potential.

        Parameters
        ----------
        charges : Tensor
            Atomic charges of all atoms (not required).
        cache : ElectricFieldCache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-wise electric field dipole potential.
        """
        return -cache.vdp

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(field={self.field})"

    def __repr__(self) -> str:
        return str(self)


def new_efield(
    field: Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ElectricField:
    """
    Create an instance of the electric field interaction.

    Parameters
    ----------
    field : Tensor
        Electric field vector consisting of the three cartesian components.
    device : torch.device | None, optional
        Device to store the tensor on. If ``None`` (default), the device is
        inferred from the `field` argument.
    dtype : torch.dtype | None, optional
        Data type of the tensor. If ``None`` (default), the data type is inferred
        from the `field` argument.

    Returns
    -------
    ElectricField
        Instance of the electric field interaction.

    Raises
    ------
    RuntimeError
        Shape of `field` is not a vector of length 3.
    """
    if field.shape != torch.Size([3]):
        raise RuntimeError("Electric field must be a vector of length 3.")

    if device is not None:
        if device != field.device:
            raise DeviceError(
                f"Passed device ({device}) and device of electric field "
                f"({field.device}) do not match."
            )

    if dtype is not None:
        if dtype != field.dtype:
            raise DtypeError(
                f"Passed dtype ({dtype}) and dtype of electric field "
                f"({field.dtype}) do not match."
            )

    return ElectricField(
        field,
        device=device if device is None else field.device,
        dtype=dtype if dtype is None else field.dtype,
    )
