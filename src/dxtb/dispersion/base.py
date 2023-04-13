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

    param: dict[str, Tensor]
    """Dispersion parameters."""

    charge: Tensor | None
    """Total charge of the system."""

    __slots__ = ["numbers", "param", "charge"]

    class Cache(ABC):
        """
        Abstract base class for the dispersion Cache.
        """

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

    def get_gradient(self, energy: Tensor, positions: Tensor) -> Tensor:
        """
        Calculates nuclear gradient of an classical energy contribution via
        PyTorch's autograd engine.

        Parameters
        ----------
        energy : Tensor
            Energy that will be differentiated.
        positions : Tensor
            Nuclear positions. Needs `requires_grad=True`.

        Returns
        -------
        Tensor
            Nuclear gradient of `energy`.

        Raises
        ------
        RuntimeError
            `positions` tensor does not have `requires_grad=True`.
        """
        if positions.requires_grad is False:
            raise RuntimeError("Position tensor needs `requires_grad=True`.")

        (gradient,) = torch.autograd.grad(
            energy, positions, grad_outputs=torch.ones_like(energy)
        )
        return gradient
