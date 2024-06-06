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
Classical repulsion energy contribution
=======================================

This module implements the classical repulsion energy term.

Note
----
The Repulsion class is constructed for geometry optimization, i.e., the atomic
numbers are set upon instantiation (`numbers` is a property), and the parameters
in the cache are created for only those atomic numbers. The positions, however,
must be supplied to the ``get_energy`` (or ``get_grad``) method.

Example
-------

.. code-block:: python

    import torch
    from dxtb import IndexHelper
    from dxtb.classical import new_repulsion
    from dxtb import GFN1_XTB

    numbers = torch.tensor([14, 1, 1, 1, 1])
    positions = torch.tensor([
        [0.00000000000000, 0.00000000000000, 0.00000000000000],
        [1.61768389755830, 1.61768389755830, -1.61768389755830],
        [-1.61768389755830, -1.61768389755830, -1.61768389755830],
        [1.61768389755830, -1.61768389755830, 1.61768389755830],
        [-1.61768389755830, 1.61768389755830, 1.61768389755830],
    ])

    rep = new_repulsion(numbers, positions, GFN1_XTB)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    cache = rep.get_cache(numbers, ihelp)
    energy = rep.get_energy(positions, cache)

    print(energy.sum(-1))  # Output: tensor(0.0303)
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs

from dxtb import IndexHelper
from dxtb._src.constants import xtb
from dxtb._src.typing import Any, Tensor

from ..base import Classical, ClassicalCache

__all__ = [
    "BaseRepulsion",
    "BaseRepulsionCache",
    "repulsion_energy",
    "repulsion_gradient",
]


class BaseRepulsionCache(ClassicalCache):
    """
    Cache for the repulsion parameters.
    """

    arep: Tensor
    """Atom-specific screening parameters."""

    zeff: Tensor
    """Effective nuclear charges."""

    kexp: Tensor
    """
    Scaling of the interatomic distance in the exponential damping function
    of the repulsion energy.
    """

    mask: Tensor
    """Mask for padding from numbers."""

    __slots__ = ["mask", "arep", "zeff", "kexp"]

    def __init__(
        self,
        mask: Tensor,
        arep: Tensor,
        zeff: Tensor,
        kexp: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            device=device if device is None else arep.device,
            dtype=dtype if dtype is None else arep.dtype,
        )
        self.mask = mask
        self.arep = arep
        self.zeff = zeff
        self.kexp = kexp


class BaseRepulsion(Classical):
    """
    Representation of the classical repulsion.
    """

    arep: Tensor
    """Atom-specific screening parameters for unique species."""

    zeff: Tensor
    """Effective nuclear charges for unique species."""

    kexp: Tensor
    """
    Scaling of the interatomic distance in the exponential damping function of
    the repulsion energy.
    """

    klight: Tensor | None
    """
    Scaling of the interatomic distance in the exponential damping function of
    the repulsion energy for light elements, i.e., H and He (only GFN2).
    """

    cutoff: float
    """Real space cutoff for repulsion interactions (default: 25.0)."""

    __slots__ = ["arep", "zeff", "kexp", "klight", "cutoff"]

    def __init__(
        self,
        arep: Tensor,
        zeff: Tensor,
        kexp: Tensor,
        klight: Tensor | None = None,
        cutoff: float = xtb.DEFAULT_REPULSION_CUTOFF,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)

        self.arep = arep.to(self.device).type(self.dtype)
        self.zeff = zeff.to(self.device).type(self.dtype)
        self.kexp = kexp.to(self.device).type(self.dtype)
        self.cutoff = cutoff

        if klight is not None:
            klight = klight.to(self.device).type(self.dtype)
        self.klight = klight

    def get_cache(self, numbers: Tensor, ihelp: IndexHelper) -> BaseRepulsionCache:
        """
        Store variables for energy and gradient calculation.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        ihelp : IndexHelper
            Helper class for indexing.

        Returns
        -------
        Repulsion.Cache
            Cache for repulsion.

        Note
        ----
        The cache of a classical contribution does not require ``positions`` as
        it only becomes useful if `numbers` remain unchanged and ``positions``
        vary, i.e., during geometry optimization.
        """
        cachvars = (numbers.detach().clone(),)

        if self.cache_is_latest(cachvars) is True:
            if not isinstance(self.cache, BaseRepulsionCache):
                raise TypeError(
                    f"Cache in {self.label} is not of type '{self.label}."
                    "Cache'. This can only happen if you manually manipulate "
                    "the cache."
                )
            return self.cache

        self._cachevars = cachvars

        # spread
        arep = ihelp.spread_uspecies_to_atom(self.arep)
        zeff = ihelp.spread_uspecies_to_atom(self.zeff)
        kexp = ihelp.spread_uspecies_to_atom(
            self.kexp.expand(torch.unique(numbers).shape)
        )

        # mask for padding
        mask = real_pairs(numbers, mask_diagonal=True)

        # Without the eps, the first backward returns nan's as described in
        # https://github.com/pytorch/pytorch/issues/2421. The second backward
        # gives nan's in gradgradcheck, because the epsilon is smaller than the
        # step size. But the actual gradient should be correct.
        eps = torch.finfo(arep.dtype).tiny
        a = torch.where(
            mask,
            torch.sqrt(arep.unsqueeze(-1) * arep.unsqueeze(-2) + eps),
            torch.tensor(0.0, **self.dd),
        )

        z = zeff.unsqueeze(-1) * zeff.unsqueeze(-2) * mask
        k = kexp.unsqueeze(-1) * kexp.new_ones(kexp.shape).unsqueeze(-2) * mask

        # GFN2 uses a different value for H and He
        if self.klight is not None:
            kmask = ~real_pairs(numbers <= 2)
            k = torch.where(kmask, k, self.klight) * mask

        self.cache = BaseRepulsionCache(mask, a, z, k)
        return self.cache

    @abstractmethod
    def get_energy(
        self, positions: Tensor, cache: BaseRepulsionCache, **kwargs: Any
    ) -> Tensor:
        """
        Get repulsion energy.

        Parameters
        ----------
        cache : Repulsion.Cache
            Cache for repulsion.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        atom_resolved : bool
            Whether to return atom-resolved energy (True) or full matrix
            (False).

        Returns
        -------
        Tensor
            (Atom-resolved) repulsion energy.
        """


def repulsion_energy(
    positions: Tensor,
    mask: Tensor,
    arep: Tensor,
    kexp: Tensor,
    zeff: Tensor,
    cutoff: float = xtb.DEFAULT_REPULSION_CUTOFF,
) -> Tensor:
    """
    Clasical repulsion energy.

    Parameters
    ----------
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    mask : Tensor
        Mask for padding.
    arep : Tensor
        Atom-specific screening parameters.
    kexp : Tensor
        Scaling of the interatomic distance in the exponential damping function
        of the repulsion energy.
    zeff : Tensor
        Effective nuclear charges.
    cutoff : float, optional
        Real-space cutoff. Defaults to `xtb.DEFAULT_REPULSION_CUTOFF`.

    Returns
    -------
    Tensor
        Atom-resolved repulsion energy
    """
    eps = torch.tensor(
        torch.finfo(positions.dtype).eps,
        dtype=positions.dtype,
        device=positions.device,
    )
    zero = torch.tensor(
        0.0,
        dtype=positions.dtype,
        device=positions.device,
    )
    _cutoff = torch.tensor(
        cutoff,
        dtype=positions.dtype,
        device=positions.device,
    )

    distances = torch.where(
        mask,
        storch.cdist(positions, positions, p=2),
        eps,
    )

    # Eq.13: R_AB ** k_f
    r1k = torch.pow(distances, kexp)

    # Eq.13: exp(- (alpha_A * alpha_B)**0.5 * R_AB ** k_f )
    exp_term = torch.exp(-arep * r1k)

    # Eq.13: repulsion energy
    return torch.where(
        mask * (distances <= _cutoff),
        storch.divide(zeff * exp_term, distances),
        zero,
    )


def repulsion_gradient(
    erep: Tensor,
    positions: Tensor,
    mask: Tensor,
    arep: Tensor,
    kexp: Tensor,
    *,
    reduced: bool = False,
) -> Tensor:
    """
    Nuclear gradient of classical repulsion energy.

    Parameters
    ----------
    erep : Tensor
        Atom-resolved repulsion energy (from `repulsion_energy`).
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    mask : Tensor
        Mask for padding.
    arep : Tensor
        Atom-specific screening parameters.
    kexp : Tensor
        Scaling of the interatomic distance in the exponential damping function
        of the repulsion energy.
    reduced : bool, optional
        Shape of the output gradient. Defaults to ``False``, which returns a
        gradient of shape `(natoms, natoms, 3)`. This is required for the custom
        backward function. If `reduced=True`, the output gradient has the
        typical shape `(natoms, 3)`.

    Returns
    -------
    Tensor
        Nuclear gradient of repulsion energy. The shape is specified by the
        `reduced` keyword argument.
    """
    eps = torch.tensor(
        torch.finfo(positions.dtype).eps,
        dtype=positions.dtype,
        device=positions.device,
    )

    distances = torch.where(
        mask,
        storch.cdist(positions, positions, p=2),
        eps,
    )

    r1k = torch.pow(distances, kexp)

    # (n_batch, n_atoms, n_atoms)
    grad = -(arep * r1k * kexp + 1.0) * erep

    # (n_batch, n_atoms, n_atoms, 3)
    rij = torch.where(
        mask.unsqueeze(-1),
        positions.unsqueeze(-2) - positions.unsqueeze(-3),
        eps,
    )

    # (n_batch, n_atoms, n_atoms)
    r2 = torch.pow(distances, 2)

    # (n_batch, n_atoms, n_atoms, 3)
    grad = torch.where(mask, storch.divide(grad, r2), eps)
    grad = grad.unsqueeze(-1) * rij

    # reduction gives (n_batch, n_atoms, 3)
    return grad if reduced is False else torch.sum(grad, dim=-2)
