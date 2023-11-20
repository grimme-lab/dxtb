"""
Base class for the integral driver.
"""
from __future__ import annotations

from abc import abstractmethod

import torch

from ..._types import Tensor, TensorLike
from ...basis import Basis, IndexHelper
from ...param import Param

# TODO: Handle Update , i.e. when new positions are given
# TODO: Handle mask via kwargs


class IntDriver(TensorLike):
    """Base class for the integral driver."""

    numbers: Tensor
    """Atomic numbers of the atoms in the system."""

    par: Param
    """Representation of parametrization of xtb model."""

    ihelp: IndexHelper
    """Helper class for indexing."""

    __slots__ = ["numbers", "par", "ihelp", "_basis", "_positions"]

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        ihelp: IndexHelper,
        *,
        basis: Basis | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)
        self.numbers = numbers
        self.par = par
        self.ihelp = ihelp
        self._basis = basis
        self._positions = None

    @property
    def basis(self) -> Basis:
        if self._basis is None:
            raise RuntimeError("Basis has not been setup.")
        return self._basis

    @basis.setter
    def basis(self, bas: Basis) -> None:
        self._basis = bas

    @abstractmethod
    def setup(
        self, numbers: Tensor, positions: Tensor, par: Param, ihelp: IndexHelper
    ) -> None:
        """
        Initialize the driver.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        par : Param
            Full `xtb` parametrization.
        ihelp : IndexHelper
            Helper for indexing.
        """
