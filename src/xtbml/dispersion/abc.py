"""
Abstract base class for dispersion models.
"""

from abc import abstractmethod

from ..interaction import Interaction
from ..typing import Tensor


class Dispersion(Interaction):
    """
    Base class for dispersion correction.

    Note:
    -----
    Dispersion should be an `Interaction` as D4 can be self-consistent.
    """

    numbers: Tensor
    """Atomic numbers of all atoms."""

    param: dict[str, float]
    """Dispersion parameters."""

    def __init__(
        self, numbers: Tensor, positions: Tensor, param: dict[str, float]
    ) -> None:
        super().__init__(positions.device, positions.dtype)
        self.numbers = numbers
        self.param = param

    @abstractmethod
    def get_energy(self, positions: Tensor, **kwargs) -> Tensor:
        """
        Get dispersion energy.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms.

        Returns
        -------
        Tensor
            Atom-resolved dispersion energy.
        """
