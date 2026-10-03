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

from dxtb import IndexHelper
from dxtb._src.typing import Any, Slicers, Tensor, override

from ..base import Interaction, InteractionCache

__all__ = ["ElectricFieldGrad", "LABEL_EFIELD_GRAD", "new_efield_grad"]


LABEL_EFIELD_GRAD = "ElectricFieldGrad"
"""Label for the 'ElectricField' interaction, coinciding with the class name."""


class ElectricFieldGradCache(InteractionCache):
    """
    Restart data for the electric field gradient interaction.
    """

    __store: Store | None
    """Storage for cache (required for culling)."""

    vat: Tensor
    """
    Atom-resolved monopolar potential from the instantaneous electric field
    gradient (shape: ``(..., nat)``).
    """

    vdp: Tensor
    """
    Atom-resolved dipolar potential from the instantaneous electric field
    gradient (shape: ``(..., nat, 3)``).
    """

    vqp: Tensor
    """
    Atom-resolved quadrupolar potential from the instantaneous electric field
    gradient (shape: ``(..., nat, 6)``).
    """

    __slots__ = ["__store", "vat", "vdp", "vqp"]

    def __init__(
        self,
        vat: Tensor,
        vdp: Tensor,
        vqp: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=device if device is None else vat.device,
            dtype=dtype if dtype is None else vat.dtype,
        )
        self.vat = vat
        self.vdp = vdp
        self.vqp = vqp
        self.__store = None

    class Store:
        """
        Storage container for cache containing ``__slots__`` before culling.
        """

        vat: Tensor
        vdp: Tensor
        vqp: Tensor

        def __init__(self, vat: Tensor, vdp: Tensor, vqp: Tensor) -> None:
            self.vat = vat
            self.vdp = vdp
            self.vqp = vqp

    def cull(self, conv: Tensor, slicers: Slicers) -> None:
        if self.__store is None:
            self.__store = self.Store(self.vat, self.vdp, self.vqp)

        slicer = slicers["atom"]
        self.vat = self.vat[tuple([~conv, *slicer])]
        self.vdp = self.vdp[tuple([~conv, *slicer, ...])]
        self.vqp = self.vqp[tuple([~conv, *slicer, ...])]

    def restore(self) -> None:
        if self.__store is None:
            raise RuntimeError("Nothing to restore. Store is empty.")

        self.vat = self.__store.vat
        self.vdp = self.__store.vdp
        self.vqp = self.__store.vqp


class ElectricFieldGrad(Interaction):
    r"""
    Instantaneous electric field gradient :math:`G_{ij}`.

    The field gradient couples to the traceless quadrupole moment
    :math:`\Theta`:

    .. math::

        E = -\frac{1}{3} \sum_{ij} G_{ij} \Theta_{ij}

    All atom-centered multipole moments contribute: the point charges and
    dipoles (about the Cartesian origin) via their second moment
    :math:`Q_{ij} = q R_i R_j + R_i \mu_j + \mu_i R_j`, with
    :math:`\Theta = \frac{3}{2} (Q - \frac{1}{3} \mathrm{tr}(Q))`, and the
    atomic quadrupole moments directly. The energy does not depend on the
    trace of :math:`G`, and the molecular quadrupole moment is obtained as
    :math:`\Theta_{ij} = -3 \, \partial E / \partial G_{ij}` (see
    :meth:`dxtb.Calculator.quadrupole`).

    .. note::

        There is no analytical nuclear gradient (``get_atom_gradient``) for
        this interaction. Forces with an electric field gradient require
        autograd.
    """

    field_grad: Tensor
    """Instantaneous electric field gradient (shape: ``(3, 3)``)."""

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

    # pylint: disable=unused-argument
    @override
    def get_cache(
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
    ) -> ElectricFieldGradCache:
        """
        Create restart data for individual interactions.

        Returns
        -------
        ElectricFieldGradCache
            Restart data for the interaction.

        Note
        ----
        If this interaction is evaluated within the `InteractionList`, `numbers`
        and `IndexHelper` will be passed as argument, too. The `**_` in the
        argument list will absorb those unnecessary arguments which are given
        as keyword-only arguments (see `Interaction.get_cache()`).
        """
        if positions is None:
            raise ValueError("Electric field gradient requires positions.")

        cachvars = (
            positions.detach().clone(),
            self.field_grad.detach().clone(),
        )

        if self.cache_is_latest(cachvars) is True:
            if not isinstance(self.cache, ElectricFieldGradCache):
                raise TypeError(
                    f"Cache in {self.label} is not of type '{self.label}."
                    "Cache'. This can only happen if you manually manipulate "
                    "the cache."
                )
            return self.cache

        self._cachevars = cachvars

        grad = self.field_grad

        # the trace of the field gradient does not couple to the traceless
        # quadrupole moment; only its symmetric part is relevant for the
        # monopole and dipole terms
        eye = torch.eye(3, device=grad.device, dtype=grad.dtype)
        grad_tl = grad - torch.diagonal(grad).sum() / 3.0 * eye
        grad_sym = 0.5 * (grad_tl + grad_tl.mT)

        # (..., nat, 3) x (3, 3) x (..., nat, 3) -> (..., nat)
        vat = -0.5 * einsum(
            "...ai,ij,...aj->...a", positions, grad_sym, positions
        )

        # (3, 3) x (..., nat, 3) -> (..., nat, 3)
        vdp = -einsum("ij,...aj->...ai", grad_sym, positions)

        # packed lower triangle of the traceless quadrupole moment (xx, yx,
        # yy, zx, zy, zz); the off-diagonal elements enter twice
        rows, cols = torch.tril_indices(3, 3, device=grad.device).unbind()
        off = (rows != cols).to(grad.dtype)
        efg = grad[rows, cols] + off * grad[cols, rows]
        vqp = (-1.0 / 3.0) * efg.expand(*positions.shape[:-1], 6)

        self.cache = ElectricFieldGradCache(vat, vdp, vqp)
        return self.cache

    @override
    def get_monopole_atom_energy(
        self, cache: ElectricFieldGradCache, qat: Tensor, **_: Any
    ) -> Tensor:
        return cache.vat * qat

    @override
    def get_dipole_atom_energy(
        self,
        cache: ElectricFieldGradCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        assert qdp is not None
        return einsum("...ax,...ax->...a", cache.vdp, qdp)

    @override
    def get_quadrupole_atom_energy(
        self,
        cache: ElectricFieldGradCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        assert qqp is not None
        return einsum("...ax,...ax->...a", cache.vqp, qqp)

    @override
    def get_monopole_atom_potential(
        self, cache: ElectricFieldGradCache, *_: Any, **__: Any
    ) -> Tensor:
        return cache.vat

    @override
    def get_dipole_atom_potential(
        self, cache: ElectricFieldGradCache, *_: Any, **__: Any
    ) -> Tensor:
        return cache.vdp

    @override
    def get_quadrupole_atom_potential(
        self, cache: ElectricFieldGradCache, *_: Any, **__: Any
    ) -> Tensor:
        return cache.vqp

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(field_grad={self.field_grad})"

    def __repr__(self) -> str:  # pragma: no cover
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
