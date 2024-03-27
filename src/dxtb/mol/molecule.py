"""
Representation of Molecule
==========================

This module contains a class for the representation of important molecular
information.

Example
-------
>>> import torch
>>> from dxtb.mol import Molecule
>>>
>>> numbers = torch.tensor([14, 1, 1, 1, 1])
>>> positions = torch.tensor([
...     [+0.00000000000000, +0.00000000000000, +0.00000000000000],
...     [+1.61768389755830, +1.61768389755830, -1.61768389755830],
...     [-1.61768389755830, -1.61768389755830, -1.61768389755830],
...     [+1.61768389755830, -1.61768389755830, +1.61768389755830],
...     [-1.61768389755830, +1.61768389755830, +1.61768389755830],
... ])
>>> mol = Mol(numbers, positions)
"""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.exceptions import DeviceError, DtypeError
from tad_mctc.typing import Any, NoReturn, Tensor, TensorLike

from ..utils import memoize

__all__ = ["Mol"]


class Mol(TensorLike):
    """
    Representation of a molecule.
    """

    __slots__ = ["_numbers", "_positions", "_charge", "_name"]

    def __init__(
        self,
        numbers: Tensor,
        positions: Tensor,
        charge: Tensor | float | int | str = 0,
        name: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)

        # check and transform all (possibly) non-tensor inputs to tensors
        charge = self._tensor(charge)

        self._numbers = numbers
        self._positions = positions
        self._charge = charge
        self._name = name

        self.checks()

    @property
    def numbers(self) -> Tensor:
        return self._numbers

    @numbers.setter
    def numbers(self, numbers: Tensor) -> None:
        self._numbers = numbers
        self.checks()

    @property
    def positions(self) -> Tensor:
        return self._positions

    @positions.setter
    def positions(self, positions: Tensor) -> None:
        self._positions = positions
        self.checks()

    @property
    def charge(self) -> Tensor:
        return self._charge

    @charge.setter
    def charge(self, charge: Tensor | float | int | str) -> None:
        self._charge = self._tensor(charge)
        self.checks()

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @memoize
    def distances(self) -> Tensor:
        """
        Calculate the distance matrix from the positions.

        .. warning::

            Memoization for this method creates a cache that stores the
            distances across all instances.

        Returns
        -------
        Tensor
            Distance matrix.
        """
        return storch.cdist(self.positions)

    def clear_cache(self) -> None:
        """Clear the cross-instance caches of all memoized methods."""
        if hasattr(self.distances, "clear"):
            self.distances.clear()

    def checks(self) -> None | NoReturn:
        """
        Check all variables for consistency.

        Raises
        ------
        RuntimeError
            Wrong device or shape errors.
        """

        # check tensor type inputs
        _check_tensor(self.numbers, min_ndim=1, max_ndim=2)
        _check_tensor(self.positions, min_ndim=2, max_ndim=3)
        _check_tensor(self.charge, min_ndim=0, max_ndim=1)

        allowed_dtypes = (torch.long, torch.int16, torch.int32, torch.int64)
        if self.numbers.dtype not in allowed_dtypes:
            raise DtypeError(
                "Dtype of atomic numbers must be one of the following to allow "
                f" indexing: '{', '.join([str(x) for x in allowed_dtypes])}', "
                f"but is '{self.numbers.dtype}'"
            )

        # check if all tensors are on the same device
        for s in self.__slots__:
            attr = getattr(self, s)
            if isinstance(attr, Tensor):
                if attr.device != self.device:
                    raise DeviceError("All tensors must be on the same device!")

        if self.numbers.shape != self.positions.shape[:-1]:
            raise RuntimeError(
                f"Shape of positions ({self.positions.shape[:-1]}) is not "
                f"consistent with atomic numbers ({self.numbers.shape})."
            )

    def _tensor(self, x: Any) -> Tensor:
        if isinstance(x, Tensor):
            return x.to(device=self.device, dtype=self.dtype)

        if isinstance(x, float):
            return torch.tensor(x, device=self.device, dtype=self.dtype)

        if isinstance(x, int):
            return torch.tensor(x, device=self.device)

        if isinstance(x, str):
            try:
                return torch.tensor(float(x), **self.dd)
            except ValueError:
                raise ValueError(f"Cannot convert string '{x}' to float")

        raise TypeError(f"Tensor-incompatible type '{type(x)}'.")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.numbers.tolist()})"

    def __repr__(self) -> str:
        return str(self)


def _check_tensor(
    x: Any,
    min_ndim: int = -1,
    max_ndim: int = 9999,
) -> None | NoReturn:
    if not isinstance(x, Tensor):
        raise TypeError(f"Variable is not a tensor but '{type(x)}'.")

    if x.ndim < min_ndim:
        raise RuntimeError(f"The tensor should not fall below {min_ndim} dimensions.")
    if x.ndim > max_ndim:
        raise RuntimeError(f"The tensor should not exceed '{max_ndim}' dimensions.")
