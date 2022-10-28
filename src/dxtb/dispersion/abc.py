"""
Abstract base class for dispersion models.
"""

from abc import ABC, abstractmethod

from ..typing import Tensor, TensorLike


class Dispersion(TensorLike):
    """
    Base class for dispersion correction.
    """

    numbers: Tensor
    """Atomic numbers of all atoms."""

    param: dict[str, float]
    """Dispersion parameters."""

    __slots__ = ["numbers", "param"]

    class Cache(ABC):
        """
        Abstract base class for the dispersion Cache.
        """

    def __init__(
        self, numbers: Tensor, positions: Tensor, param: dict[str, float]
    ) -> None:
        super().__init__(positions.device, positions.dtype)
        self.numbers = numbers
        self.param = param

    @abstractmethod
    def get_cache(self, numbers: Tensor) -> "Cache":
        """
        Store variables for energy calculation.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of all atoms.

        Returns
        -------
        Cache
            Cache class for storage of variables.
        """

    @abstractmethod
    def get_energy(self, positions: Tensor, cache: "Cache") -> Tensor:
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
