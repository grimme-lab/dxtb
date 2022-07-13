import torch

from .abc import Interaction
from ..basis import IndexHelper
from ..typing import Tensor, List


class InteractionList(Interaction):
    """
    List of interactions.
    """

    def __init__(self, interactions: List[Interaction]):
        Interaction.__init__(self)
        self.interactions = interactions

    def get_cache(self, numbers: Tensor, positions: Tensor, ihelp: IndexHelper) -> "Cache":
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
        Cache
            Restart data for the interaction.
        """

        return {
            interaction.label: interaction.get_cache(numbers, positions, ihelp)
            for interaction in self.interactions
        }

    def get_potential(
        self, charges: Tensor, ihelp: IndexHelper, cache: "Cache"
    ) -> Tensor:
        """
        Compute the potential for a list of interactions.

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
            Potential vector for each orbital partial charge.
        """

        return torch.stack(
            [
                interaction.get_potential(charges, cache[interaction.label])
                for interaction in self.interactions
            ]
        ).sum(dim=0)

    def get_energy(self, charges: Tensor, ihelp: IndexHelper, cache: "Cache") -> Tensor:
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

        return torch.stack(
            [
                interaction.get_energy(charges, cache[interaction.label])
                for interaction in self.interactions
            ]
        ).sum(dim=0)
