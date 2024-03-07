"""
Abstract base class for dispersion models.
"""

from __future__ import annotations

from abc import abstractmethod

import torch

from .._types import Tensor
from ..classical import Classical


class Dispersion(Classical):
    """
    Base class for dispersion correction.
    """

    numbers: Tensor
    """Atomic numbers for all atoms in the system."""

    param: dict[str, Tensor]
    """Dispersion parameters."""

    charge: Tensor | None
    """Total charge of the system."""

    __slots__ = ["numbers", "param", "charge"]

    def __init__(
        self,
        numbers: Tensor,
        param: dict[str, Tensor],
        charge: Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.numbers = numbers
        self.param = param
        self.charge = charge

    @abstractmethod
    def get_cache(self, numbers: Tensor) -> Classical.Cache:
        """
        Store variables for energy calculation.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.

        Returns
        -------
        Cache
            Cache class for storage of variables.
        """
