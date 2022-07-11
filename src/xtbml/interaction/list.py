import torch

from .abc import Interaction
from ..basis import IndexHelper
from ..typing import Tensor, List


class InteractionList(Interaction):
    """
    List of interactions.
    """

    def __init__(self, interactions: List[Interaction]):
        self.interactions = interactions

    def get_potential(self, charges: Tensor, ihelp: IndexHelper) -> Tensor:
        """
        Compute the potential for a list of interactions.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Potential vector for each orbital partial charge.
        """

        return torch.stack(
            [interaction.get_potential(charges) for interaction in self.interactions]
        ).sum(dim=0)

    def get_energy(self, charges: Tensor, ihelp: IndexHelper) -> Tensor:
        """
        Compute the energy for a list of interactions.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Energy vector for each orbital partial charge.
        """

        return torch.stack(
            [interaction.get_energy(charges) for interaction in self.interactions]
        ).sum(dim=0)
