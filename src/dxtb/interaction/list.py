"""
Container for interactions.
"""
from __future__ import annotations

import torch

from .._types import Slicers, Tensor, TensorOrTensors
from ..basis import IndexHelper
from .base import Interaction
from .potential import Potential


class InteractionList(Interaction):
    """
    List of interactions.
    """

    class Cache(dict):
        """
        List of interaction caches.
        """

        __slots__ = ()

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

    def __init__(self, *interactions: Interaction | None) -> None:
        super().__init__(torch.device("cpu"), torch.float)
        self.interactions = [
            interaction for interaction in interactions if interaction is not None
        ]

    def get_cache(
        self, numbers: Tensor, positions: Tensor, ihelp: IndexHelper
    ) -> InteractionList.Cache:
        """
        Create restart data for individual interactions.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers.
        positions : Tensor
            Cartesian coordinates.
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
                for interaction in self.interactions
            }
        )
        return cache

    def get_potential(
        self, charges: Tensor, cache: InteractionList.Cache, ihelp: IndexHelper
    ) -> Potential:
        """
        Compute the potential for a list of interactions.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges.
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
            torch.zeros_like(charges), vdipole=None, vquad=None, batched=ihelp.batched
        )

        # exit with empty potential if no interactions present
        if len(self.interactions) <= 0:
            return pot

        # add up potentials from all interactions
        for interaction in self.interactions:
            p = interaction.get_potential(charges, cache[interaction.label], ihelp)
            pot += p

        return pot

    def get_energy_as_dict(
        self, charges: Tensor, cache: Cache, ihelp: IndexHelper
    ) -> dict[str, Tensor]:
        """
        Compute the energy for a list of interactions.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Energy vector for each orbital partial charge.
        """
        if len(self.interactions) <= 0:
            return {"none": torch.zeros_like(charges)}

        return {
            interaction.label: interaction.get_energy(
                charges, cache[interaction.label], ihelp
            )
            for interaction in self.interactions
        }

    def get_energy(self, charges: Tensor, cache: Cache, ihelp: IndexHelper) -> Tensor:
        """
        Compute the energy for a list of interactions.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-resolved energy vector for orbital partial charges.
        """
        if len(self.interactions) <= 0:
            return ihelp.reduce_orbital_to_atom(torch.zeros_like(charges))

        return torch.stack(
            [
                interaction.get_energy(charges, cache[interaction.label], ihelp)
                for interaction in self.interactions
            ]
        ).sum(dim=0)

    def get_gradient(
        self,
        charges: Tensor,
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
        charges : Tensor
            Orbital-resolved partial charges.
        positions : Tensor
            Cartesian coordinates.
        cache : InteractionList.Cache
            Restart data for the interaction.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Nuclear gradient of all interactions.
        """
        if len(self.interactions) <= 0:
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
                for interaction in self.interactions
            ]
        ).sum(dim=0)
