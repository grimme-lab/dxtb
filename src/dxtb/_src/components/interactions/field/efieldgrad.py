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
External Fields: Field Gradient
===============================

Interaction of the charge density with external electric field gradient.
"""

from __future__ import annotations

import torch
from tad_mctc.exceptions import DeviceError, DtypeError
from tad_mctc.math import einsum

from dxtb._src.typing import Any, Tensor, TensorLike

from ..base import Interaction, InteractionCache

__all__ = ["ElectricFieldGrad", "LABEL_EFIELD_GRAD", "new_efield_grad"]


LABEL_EFIELD_GRAD = "ElectricFieldGrad"
"""Label for the 'ElectricField' interaction, coinciding with the class name."""


class ElectricFieldCache(InteractionCache, TensorLike):
    """
    Restart data for the electric field interaction.

    Note
    ----
    This cache is not culled, and hence, does not contain a `Store`.
    """

    efg: Tensor
    """Reshaped electric field gradient."""

    __slots__ = ["efg"]

    def __init__(
        self,
        efg: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            device=device if device is None else efg.device,
            dtype=dtype if dtype is None else efg.dtype,
        )
        self.efg = efg


class ElectricFieldGrad(Interaction):
    """
    Electric field gradient.
    """

    field_grad: Tensor
    """Electric field gradient."""

    __slots__ = ["field_grad"]

    def __init__(
        self,
        field_grad: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            device=device if device is None else field_grad.device,
            dtype=dtype if dtype is None else field_grad.dtype,
        )
        self.field_grad = field_grad

    def get_cache(self, **_: Any) -> ElectricFieldCache:
        """
        Create restart data for individual interactions.

        Returns
        -------
        ElectricFieldCache
            Restart data for the interaction.

        Note
        ----
        Here, this is only a dummy.
        """
        cachvars = (self.field_grad.detach().clone(),)

        if self.cache_is_latest(cachvars) is True:
            if not isinstance(self.cache, ElectricFieldCache):
                raise TypeError(
                    f"Cache in {self.label} is not of type '{self.label}."
                    "Cache'. This can only happen if you manually manipulate "
                    "the cache."
                )
            return self.cache

        self._cachevars = cachvars

        efg = self.field_grad[torch.tril_indices(3, 3).unbind()]
        self.cache = ElectricFieldCache(efg)

        return self.cache

    # TODO: This is probably not correct...
    def get_quadrupole_energy(
        self, charges: Tensor, cache: ElectricFieldCache
    ) -> Tensor:
        """
        Calculate the quadrupolar contribution of the electric field energy.

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

        # equivalent: torch.sum(-cache.vqp * charges, dim=-1)
        return 0.5 * einsum("...x,...ix->...i", cache.efg, charges)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(field_grad={self.field_grad})"

    def __repr__(self) -> str:
        return str(self)


def new_efield_grad(
    field_grad: Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ElectricFieldGrad:
    """
    Create an instance of the electric field gradient interaction.

    Parameters
    ----------
    field_grad : Tensor
        Electric field gradient consisting of the 3x3 cartesian components.
    device : torch.device | None, optional
        Device to store the tensor on. If ``None`` (default), the device is
        inferred from the `field` argument.
    dtype : torch.dtype | None, optional
        Data type of the tensor. If ``None`` (default), the data type is inferred
        from the `field` argument.

    Returns
    -------
    ElectricFieldGrad
        Instance of the electric field gradient interaction.

    Raises
    ------
    RuntimeError
        Shape of `field_grad` is not a 3x3 matrix.
    """
    if field_grad.shape != torch.Size((3, 3)):
        raise RuntimeError("Electric field gradient must be a 3 by 3 matrix.")

    if device is not None:
        if device != field_grad.device:
            raise DeviceError(
                f"Passed device ({device}) and device of electric field "
                f"gradient ({field_grad.device}) do not match."
            )

    if dtype is not None:
        if dtype != field_grad.dtype:
            raise DtypeError(
                f"Passed dtype ({dtype}) and dtype of electric field "
                f"gradient ({field_grad.dtype}) do not match."
            )

    return ElectricFieldGrad(
        field_grad,
        device=device if device is None else field_grad.device,
        dtype=dtype if dtype is None else field_grad.dtype,
    )
