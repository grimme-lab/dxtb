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
Container for interactions.
"""

from __future__ import annotations

import torch

from dxtb.basis import IndexHelper
from dxtb.typing import Any, Literal, Tensor, TensorOrTensors, overload, override

from ...components.list import ComponentList, _docstring_reset, _docstring_update
from .base import Interaction
from .container import Charges, Potential
from .coulomb.secondorder import ES2, LABEL_ES2
from .coulomb.thirdorder import ES3, LABEL_ES3
from .external.field import LABEL_EFIELD, ElectricField
from .external.fieldgrad import LABEL_EFIELD_GRAD, ElectricFieldGrad

__all__ = ["InteractionList"]


class InteractionList(ComponentList[Interaction]):
    """
    List of interactions.
    """

    @override
    def get_energy(
        self,
        charges: Charges | Tensor,
        cache: ComponentList.Cache,
        ihelp: IndexHelper,
    ) -> Tensor:
        """
        Compute the energy for a list of interactions.

        Parameters
        ----------
        charges : Charges | Tensor
            Collection of charges. Monopolar partial charges are
            orbital-resolved.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : ComponentList.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-resolved energy vector for orbital partial charges.
        """
        if isinstance(charges, Tensor):
            charges = Charges(mono=charges)

        if len(self.components) <= 0:
            return ihelp.reduce_orbital_to_atom(torch.zeros_like(charges.mono))

        return torch.stack(
            [
                interaction.get_energy(charges, cache[interaction.label], ihelp)
                for interaction in self.components
            ]
        ).sum(dim=0)

    def get_energy_as_dict(
        self, charges: Charges, cache: ComponentList.Cache, ihelp: IndexHelper
    ) -> dict[str, Tensor]:
        """
        Compute the energy for a list of interactions.

        Parameters
        ----------
        charges : Charges
            Collection of charges. Monopolar partial charges are
            orbital-resolved.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : ComponentList.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Energy vector for each orbital partial charge.
        """
        if len(self.components) <= 0:
            return {"none": torch.zeros_like(charges.mono)}

        return {
            interaction.label: interaction.get_energy(
                charges, cache[interaction.label], ihelp
            )
            for interaction in self.components
        }

    @override
    def get_gradient(
        self,
        charges: Charges,
        positions: Tensor,
        cache: InteractionList.Cache,
        ihelp: IndexHelper,
        grad_outputs: TensorOrTensors | None = None,
        retain_graph: bool | None = True,
        create_graph: bool | None = None,
    ) -> Tensor:
        """
        Calculate gradient for a list of interactions.

        Parameters
        ----------
        charges : Charges
            Collection of charges. Monopolar partial charges are
            orbital-resolved.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        cache : InteractionList.Cache
            Restart data for the interaction.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Nuclear gradient of all interactions.
        """
        if len(self.components) <= 0:
            return torch.zeros_like(positions)

        return torch.stack(
            [
                interaction.get_gradient(
                    charges,
                    positions,
                    cache[interaction.label],
                    ihelp,
                    grad_outputs=grad_outputs,
                    retain_graph=retain_graph,
                    create_graph=create_graph,
                )
                for interaction in self.components
            ]
        ).sum(dim=0)

    @override
    def get_cache(
        self, numbers: Tensor, positions: Tensor, ihelp: IndexHelper
    ) -> InteractionList.Cache:
        """
        Create restart data for individual interactions.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        ihelp: IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        InteractionList.Cache
            Restart data for the interactions.
        """
        cache = self.Cache()
        cache.update(
            **{
                interaction.label: interaction.get_cache(
                    numbers=numbers, positions=positions, ihelp=ihelp
                )
                for interaction in self.components
            }
        )
        return cache

    def get_potential(
        self, charges: Charges, cache: InteractionList.Cache, ihelp: IndexHelper
    ) -> Potential:
        """
        Compute the potential for a list of interactions.

        Parameters
        ----------
        charges : Charges
            Collection of charges. Monopolar partial charges are
            orbital-resolved.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : InteractionList.Cache
            Restart data for the interactions.

        Returns
        -------
        Tensor
            Potential vector for each orbital partial charge.
        """

        # create empty potential
        pot = Potential(
            torch.zeros_like(charges.mono),
            dipole=None,
            quad=None,
            batch_mode=ihelp.batch_mode,
        )

        # exit with empty potential if no interactions present
        if len(self.components) <= 0:
            return pot

        # add up potentials from all interactions
        for interaction in self.components:
            p = interaction.get_potential(charges, cache[interaction.label], ihelp)
            pot += p

        return pot

    ###########################################################################

    @overload
    def get_interaction(self, name: Literal["ElectricField"]) -> ElectricField: ...

    @overload
    def get_interaction(
        self, name: Literal["ElectricFieldGrad"]
    ) -> ElectricFieldGrad: ...

    @overload
    def get_interaction(self, name: Literal["ES2"]) -> ES2: ...

    @overload
    def get_interaction(self, name: Literal["ES3"]) -> ES3: ...

    @override  # generic implementation for typing
    def get_interaction(self, name: str) -> Interaction:
        return super().get_interaction(name)

    ###########################################################################

    @_docstring_reset
    def reset_efield(self) -> Interaction:
        """Reset tensor attributes to a detached clone of the current state."""
        return self.reset(LABEL_EFIELD)

    @_docstring_reset
    def reset_efield_grad(self) -> Interaction:
        """Reset tensor attributes to a detached clone of the current state."""
        return self.reset(LABEL_EFIELD_GRAD)

    @_docstring_reset
    def reset_es2(self) -> Interaction:
        """Reset tensor attributes to a detached clone of the current state."""
        return self.reset(LABEL_ES2)

    @_docstring_reset
    def reset_es3(self) -> Interaction:
        """Reset tensor attributes to a detached clone of the current state."""
        return self.reset(LABEL_ES3)

    ###########################################################################

    @_docstring_update
    def update_efield(
        self,
        *,
        field: Tensor | None = None,
    ) -> Interaction:
        return self.update(LABEL_EFIELD, field=field)

    @_docstring_update
    def update_efield_grad(
        self,
        *,
        field_grad: Tensor | None = None,
    ) -> Interaction:
        return self.update(LABEL_EFIELD_GRAD, field_grad=field_grad)

    @_docstring_update
    def update_es2(self, **kwargs: Any) -> Interaction:
        return self.update(LABEL_ES2, **kwargs)

    @_docstring_update
    def update_es3(self, **kwargs: Any) -> Interaction:
        return self.update(LABEL_ES3, **kwargs)
