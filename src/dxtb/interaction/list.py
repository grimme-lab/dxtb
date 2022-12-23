"""
Container for interactions.
"""
from __future__ import annotations

import torch

from .._types import Tensor
from ..basis import IndexHelper
from .base import Interaction


class InteractionList(Interaction):
    """
    List of interactions.
    """

    class Cache(Interaction.Cache, dict):
        """
        List of interaction caches.
        """

        __slots__ = ()

    def __init__(self, *interactions: Interaction | None) -> None:
        super().__init__(torch.device("cpu"), torch.float)
        self.interactions = [
            interaction for interaction in interactions if interaction is not None
        ]

    def get_cache(
        self, numbers: Tensor, positions: Tensor, ihelp: IndexHelper
    ) -> Cache:
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
        Interaction.Cache
            Restart data for the interaction.
        """

        cache = self.Cache()
        cache.update(
            **{
                interaction.label: interaction.get_cache(
                    numbers, positions, ihelp=ihelp
                )
                for interaction in self.interactions
            }
        )
        return cache

    def get_potential(
        self, charges: Tensor, cache: Cache, ihelp: IndexHelper
    ) -> Tensor:
        """
        Compute the potential for a list of interactions.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : Interaction.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Potential vector for each orbital partial charge.
        """

        return (
            torch.stack(
                [
                    interaction.get_potential(charges, cache[interaction.label], ihelp)
                    for interaction in self.interactions
                ]
            ).sum(dim=0)
            if len(self.interactions) > 0
            else torch.zeros_like(charges)
        )

    def get_energy(self, charges: Tensor, cache: Cache, ihelp: IndexHelper) -> Tensor:
        """
        Compute the energy for a list of interactions.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : Interaction.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Energy vector for each orbital partial charge.
        """

        return (
            torch.stack(
                [
                    interaction.get_energy(charges, cache[interaction.label], ihelp)
                    for interaction in self.interactions
                ]
            ).sum(dim=0)
            if len(self.interactions) > 0
            else charges.new_zeros(())
        )
