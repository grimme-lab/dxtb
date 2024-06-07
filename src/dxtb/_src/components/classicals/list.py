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
Components: List
================

Container for classical contributions.
"""

from __future__ import annotations

import torch

from dxtb._src.timing import timer
from dxtb._src.typing import TYPE_CHECKING, Any, Literal, Tensor, overload, override

from ..list import ComponentList, ComponentListCache
from ..utils import _docstring_reset, _docstring_update
from .base import Classical
from .halogen import LABEL_HALOGEN, Halogen
from .repulsion import LABEL_REPULSION, Repulsion

if TYPE_CHECKING:
    from dxtb import IndexHelper

__all__ = ["ClassicalList", "ClassicalListCache"]


class ClassicalListCache(ComponentListCache):
    """
    Cache for classical contributions.
    """


class ClassicalList(ComponentList[Classical]):
    """
    List of classical contributions.
    """

    @override
    def get_cache(self, numbers: Tensor, ihelp: IndexHelper) -> ClassicalListCache:
        """
        Create restart data for individual classical contributions.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        ihelp: IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        ClassicalListCache
            Restart data for the classicals.
        """
        cache = ClassicalListCache()

        d = {}
        for classical in self.components:
            timer.start(classical.label, parent_uid="Classicals")
            d[classical.label] = classical.get_cache(numbers=numbers, ihelp=ihelp)
            timer.stop(classical.label)

        cache.update(**d)
        return cache

    @override
    def get_energy(
        self, positions: Tensor, cache: ClassicalListCache
    ) -> dict[str, Tensor]:
        """
        Compute the energy for a list of classicals.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        cache : ClassicalListCache
            Restart data for the classical contribution.

        Returns
        -------
        dict[str, Tensor]
            Energy vectors of all classical contributions.
        """
        if len(self.components) <= 0:
            return {"none": positions.new_zeros(positions.shape[:-1])}

        energies = {}
        for classical in self.components:
            timer.start(classical.label, parent_uid="Classicals")
            energies[classical.label] = classical.get_energy(
                positions, cache[classical.label]
            )
            timer.stop(classical.label)

        return energies

    @override
    def get_gradient(
        self, energy: dict[str, Tensor], positions: Tensor
    ) -> dict[str, Tensor]:
        """
        Calculate gradient for a list of classicals.

        Parameters
        ----------
        energy : dict[str, Tensor]
            Energies of all classical contributions that will be differentiated.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).

        Returns
        -------
        dict[str, Tensor]
            Nuclear gradients of all classical contributions.
        """
        if len(self.components) <= 0:
            return {"none": torch.zeros_like(positions)}

        gradients = {}
        for classical in self.components:
            timer.start(f"{classical.label} Gradient")
            gradients[classical.label] = classical.get_gradient(
                energy[classical.label], positions
            )
            timer.stop(f"{classical.label} Gradient")

        return gradients

    ###########################################################################

    @overload
    def get_interaction(self, name: Literal["Halogen"]) -> Halogen: ...

    @overload
    def get_interaction(self, name: Literal["Repulsion"]) -> Repulsion: ...

    @override  # generic implementation for typing
    def get_interaction(self, name: str) -> Classical:
        return super().get_interaction(name)

    ###########################################################################

    @_docstring_reset
    def reset_halogen(self) -> Classical:
        """Reset tensor attributes to a detached clone of the current state."""
        return self.reset(LABEL_HALOGEN)

    @_docstring_reset
    def reset_repulsion(self) -> Classical:
        """Reset tensor attributes to a detached clone of the current state."""
        return self.reset(LABEL_REPULSION)

    ###########################################################################

    @_docstring_update
    def update_halogen(self, **kwargs: Any) -> Classical:
        return self.update(LABEL_HALOGEN, **kwargs)

    @_docstring_update
    def update_repulsion(self, **kwargs: Any) -> Classical:
        """"""
        return self.update(LABEL_REPULSION, **kwargs)
