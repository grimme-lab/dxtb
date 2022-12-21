"""
Abstract base class for dispersion models.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .._types import Tensor, TensorLike


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
        self,
        numbers: Tensor,
        param: dict[str, float],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.numbers = numbers
        self.param = param

    @abstractmethod
    def get_cache(self, numbers: Tensor) -> Cache:
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
    def get_energy(self, positions: Tensor, cache: Cache) -> Tensor:
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
