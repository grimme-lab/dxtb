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

from dxtb import IndexHelper
from dxtb._src.typing import (
    Any,
    Literal,
    Slicers,
    Tensor,
    TensorOrTensors,
    overload,
    override,
)

from ..list import ComponentList, ComponentListCache
from ..utils import _docstring_reset, _docstring_update
from .base import Interaction
from .container import Charges, Potential
from .coulomb.secondorder import ES2, LABEL_ES2
from .coulomb.thirdorder import ES3, LABEL_ES3
from .dispersion.d4sc import LABEL_DISPERSIOND4SC, DispersionD4SC
from .field.efield import LABEL_EFIELD, ElectricField
from .field.efieldgrad import LABEL_EFIELD_GRAD, ElectricFieldGrad
from .spin.spinpolarisation import LABEL_SPINPOLARISATION, SpinPolarisation

__all__ = ["InteractionList", "InteractionListCache"]


def _has_spin_dim(mono: Tensor) -> bool:
    """Check whether the monopolar charges carry a spin dimension.

    In unrestricted (UHF) mode, ``Charges.mono`` has shape ``(..., nao, 2)``
    where channel 0 is the total charge and channel 1 is the magnetization.
    """
    return mono.ndim >= 2 and mono.shape[-1] == 2


class InteractionListCache(ComponentListCache):
    """
    Restart data for individual interactions, extended by subclasses as
    needed.
    """

    def cull(self, conv: Tensor, slicers: Slicers) -> None:
        """
        Cull all interaction caches.

        Parameters
        ----------
        conv : Tensor
            Mask of converged systems.
        """
        for cache in self.values():
            cache.cull(conv, slicers)

    def restore(self) -> None:
        """
        Restore all interaction caches.
        """
        for cache in self.values():
            cache.restore()


class InteractionList(ComponentList[Interaction]):
    """
    List of interactions.
    """

    @override
    def get_energy(
        self,
        charges: Charges | Tensor,
        cache: InteractionListCache,
        ihelp: IndexHelper,
    ) -> Tensor:
        """
        Compute the energy for a list of interactions.

        In unrestricted (UHF) mode, ``charges.mono`` has shape
        ``(..., nao, 2)`` (charge/magnetization). Non-spin interactions
        receive only the charge channel; :class:`SpinPolarisation` receives
        shell-resolved charge/magnetization stacked along the last dim.

        Parameters
        ----------
        charges : Charges | Tensor
            Collection of charges. Monopolar partial charges are
            orbital-resolved.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : InteractionListCache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-resolved energy vector for orbital partial charges.
        """
        if isinstance(charges, Tensor):
            charges = Charges(mono=charges)

        has_spin = _has_spin_dim(charges.mono)

        # In UHF mode, create a total-charge-only Charges for non-spin
        # interactions and stacked shell charges for SpinPolarisation.
        if has_spin:
            charges_total = Charges(
                mono=charges.mono[..., 0],
                dipole=charges.dipole,
                quad=charges.quad,
                batch_mode=charges.batch_mode,
            )
        else:
            charges_total = charges

        if len(self.components) <= 0:
            return ihelp.reduce_orbital_to_atom(
                torch.zeros_like(charges_total.mono)
            )

        energies: list[Tensor] = []
        for interaction in self.components:
            if has_spin and isinstance(interaction, SpinPolarisation):
                # Build shell-resolved [nsh, 2] for SpinPolarisation
                qsh_charge = ihelp.reduce_orbital_to_shell(charges.mono[..., 0])
                qsh_mag = ihelp.reduce_orbital_to_shell(charges.mono[..., -1])
                qsh = torch.stack([qsh_charge, qsh_mag], dim=-1)

                esh = interaction.get_monopole_shell_energy(
                    cache[interaction.label], qsh
                )
                energies.append(ihelp.reduce_shell_to_atom(esh))
            else:
                energies.append(
                    interaction.get_energy(
                        cache[interaction.label], charges_total, ihelp
                    )
                )

        return torch.stack(energies).sum(dim=0)

    def get_energy_as_dict(
        self, charges: Charges, cache: InteractionListCache, ihelp: IndexHelper
    ) -> dict[str, Tensor]:
        """
        Compute the energy for a list of interactions, returned as a dict.

        Handles spin-resolved charges analogously to :meth:`get_energy`.
        """
        has_spin = _has_spin_dim(charges.mono)

        if has_spin:
            charges_total = Charges(
                mono=charges.mono[..., 0],
                dipole=charges.dipole,
                quad=charges.quad,
                batch_mode=charges.batch_mode,
            )
        else:
            charges_total = charges

        if len(self.components) <= 0:
            return {"none": torch.zeros_like(charges_total.mono)}

        result: dict[str, Tensor] = {}
        for interaction in self.components:
            if has_spin and isinstance(interaction, SpinPolarisation):
                qsh_charge = ihelp.reduce_orbital_to_shell(charges.mono[..., 0])
                qsh_mag = ihelp.reduce_orbital_to_shell(charges.mono[..., -1])
                qsh = torch.stack([qsh_charge, qsh_mag], dim=-1)
                esh = interaction.get_monopole_shell_energy(
                    cache[interaction.label], qsh
                )
                result[interaction.label] = ihelp.reduce_shell_to_atom(esh)
            else:
                result[interaction.label] = interaction.get_energy(
                    cache[interaction.label], charges_total, ihelp
                )
        return result

    @override
    def get_gradient(
        self,
        charges: Charges,
        positions: Tensor,
        cache: InteractionListCache,
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
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        cache : InteractionListCache
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
    ) -> InteractionListCache:
        """
        Create restart data for individual interactions.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        ihelp: IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        InteractionListCache
            Restart data for the interactions.
        """
        cache = InteractionListCache()
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
        self, cache: InteractionListCache, charges: Charges, ihelp: IndexHelper
    ) -> Potential:
        """
        Compute the potential for a list of interactions.

        In unrestricted (UHF) mode, ``charges.mono`` has shape
        ``(..., nao, 2)`` (charge/magnetization). Non-spin interactions are
        evaluated on the charge channel only. The :class:`SpinPolarisation`
        interaction contributes a magnetization potential that is placed into
        channel 1 of the returned ``Potential.mono`` tensor.

        Parameters
        ----------
        cache : InteractionListCache
            Restart data for the interactions.
        charges : Charges
            Collection of charges. Monopolar partial charges are
            orbital-resolved.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Potential
            Potential container. In UHF mode ``mono`` has shape
            ``(..., nao, 2)`` (charge potential, magnetization potential).
        """
        has_spin = _has_spin_dim(charges.mono)

        if has_spin:
            charges_total = Charges(
                mono=charges.mono[..., 0],
                dipole=charges.dipole,
                quad=charges.quad,
                batch_mode=charges.batch_mode,
            )
        else:
            charges_total = charges

        # Create empty potential for non-spin (charge) contributions
        pot = Potential(
            torch.zeros_like(charges_total.mono),
            dipole=None,
            quad=None,
            batch_mode=ihelp.batch_mode,
        )

        # exit with empty potential if no interactions present
        if len(self.components) <= 0:
            if has_spin:
                pot.mono = torch.stack(
                    [pot.mono, torch.zeros_like(pot.mono)], dim=-1
                )
            return pot

        # Magnetization potential accumulator (only in UHF mode)
        v_spin = torch.zeros_like(charges_total.mono) if has_spin else None

        for interaction in self.components:
            if has_spin and isinstance(interaction, SpinPolarisation):
                # Build shell-resolved [nsh, 2] for SpinPolarisation
                qsh_charge = ihelp.reduce_orbital_to_shell(charges.mono[..., 0])
                qsh_mag = ihelp.reduce_orbital_to_shell(charges.mono[..., -1])
                qsh = torch.stack([qsh_charge, qsh_mag], dim=-1)

                # Get shell-level spin potential: shape [..., nsh, 2]
                vsh_spin = interaction.get_monopole_shell_potential(
                    cache[interaction.label], qsh
                )
                # Spread magnetization channel to orbital resolution
                assert v_spin is not None
                v_spin = v_spin + ihelp.spread_shell_to_orbital(
                    vsh_spin[..., -1]
                )
            else:
                p = interaction.get_potential(
                    cache[interaction.label], charges_total, ihelp
                )
                pot += p

        if has_spin:
            assert v_spin is not None
            assert pot.mono is not None
            combined = torch.stack([pot.mono, v_spin], dim=-1)
            return Potential(
                mono=combined,
                dipole=pot.dipole,
                quad=pot.quad,
                batch_mode=ihelp.batch_mode,
            )

        return pot

    ###########################################################################

    @overload
    def get_interaction(
        self, name: Literal["DispersionD4SC"]
    ) -> DispersionD4SC: ...

    @overload
    def get_interaction(
        self, name: Literal["ElectricField"]
    ) -> ElectricField: ...

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
    def reset_d4sc(self) -> Interaction:
        """Reset tensor attributes to a detached clone of the current state."""
        return self.reset(LABEL_DISPERSIOND4SC)

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

    @_docstring_reset
    def reset_spinpolarisation(self) -> Interaction:
        """Reset tensor attributes to a detached clone of the current state."""
        return self.reset(LABEL_SPINPOLARISATION)

    ###########################################################################

    @_docstring_update
    def update_d4sc(self, **kwargs: Any) -> Interaction:
        return self.update(LABEL_DISPERSIOND4SC, **kwargs)

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

    @_docstring_update
    def update_spinpolarisation(self, **kwargs: Any) -> Interaction:
        return super().update(LABEL_SPINPOLARISATION, **kwargs)
