"""
Representation of Molecule
==========================

This module contains a class for the representation of important molecular
information.
"""
from __future__ import annotations

import torch

from .._types import Tensor, TensorLike

__all__ = ["Mol"]


class Mol(TensorLike):
    """
    Representation of a molecule.
    """

    __slots__ = ["_numbers", "_positions"]

    def __init__(
        self,
        numbers: Tensor,
        positions: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)

        self._positions = positions
        self._numbers = numbers

    @property
    def numbers(self) -> Tensor:
        return self._numbers

    @numbers.setter
    def numbers(self, numbers: Tensor) -> None:
        self._numbers = numbers

    @property
    def positions(self) -> Tensor:
        return self._positions

    @positions.setter
    def positions(self, positions: Tensor) -> None:
        self._positions = positions
