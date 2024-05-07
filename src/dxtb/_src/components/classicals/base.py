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
Classical contributions (ABC)
=============================

This module contains the abstract base class for all classical (i.e., non-
selfconsistent or non-density-dependent) energy terms.

Every contribution contains a class:`dxtb.components.ComponentCache` that holds
position-independent variables. Therefore, the positions must always be
supplied to the ``get_energy`` (or ``get_grad``) method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from dxtb._src.typing import TYPE_CHECKING, Any, Tensor

from ..base import Component, ComponentCache

if TYPE_CHECKING:
    from dxtb import IndexHelper


__all__ = ["ClassicalABC", "Classical", "ClassicalCache"]


class ClassicalABC(ABC):
    """
    Abstract base class for calculation of classical contributions.
    """

    @abstractmethod
    def get_cache(self, numbers: Tensor, ihelp: IndexHelper) -> ComponentCache:
        """
        Store variables for energy calculation.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        ihelp : IndexHelper
            Helper class for indexing.

        Returns
        -------
        Cache
            Cache class for storage of variables.

        Note
        ----
        The cache of a classical contribution does not require ``positions`` as
        it only becomes useful if ``numbers`` remain unchanged and ``positions``
        vary, i.e., during geometry optimization.
        """

    @abstractmethod
    def get_energy(
        self, positions: Tensor, cache: ComponentCache, **kwargs: Any
    ) -> Tensor:
        """
        Obtain energy of the contribution.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        cache : Cache
            Cache for the parameters.

        Returns
        -------
        Tensor
            Atomwise energy contributions.
        """


class ClassicalCache(ComponentCache):
    """
    Restart data for individual classical contributions.
    """

    __slots__: list[str] = []


class Classical(ClassicalABC, Component):
    """
    Base class for calculation of classical contributions.
    """

    label: str
    """Label for the classical contribution."""

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)

    def get_gradient(
        self, energy: Tensor, positions: Tensor, grad_outputs: Tensor | None = None
    ) -> Tensor:
        """
        Calculates nuclear gradient of a classical energy contribution via
        PyTorch's autograd engine.

        Parameters
        ----------
        energy : Tensor
            Energy that will be differentiated.
        positions : Tensor
            Nuclear positions. Needs ``requires_grad=True``.
        grad_outputs : Tensor | None, optional
            Vector in the vector-Jacobian product. If ``None``, the vector is
            initialized to ones.

        Returns
        -------
        Tensor
            Nuclear gradient of ``energy``.

        Raises
        ------
        RuntimeError
            ``positions`` tensor does not have ``requires_grad=True``.
        """
        if positions.requires_grad is False:
            raise RuntimeError("Position tensor needs ``requires_grad=True``.")

        # avoid autograd call if energy is zero (autograd fails anyway)
        if torch.equal(energy, torch.zeros_like(energy)):
            return torch.zeros_like(positions)

        g = torch.ones_like(energy) if grad_outputs is None else grad_outputs
        (gradient,) = torch.autograd.grad(energy, positions, grad_outputs=g)
        return gradient
