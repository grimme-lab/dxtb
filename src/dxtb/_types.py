"""
Type annotations for this project.
"""
# pylint: disable=unused-import
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, TypedDict, overload

import torch
from torch import Tensor

from .constants import defaults

# Python 3.11
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# Python 3.10
if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

# Python 3.9 + "from __future__ import annotations"
if sys.version_info >= (3, 9):
    # starting with Python 3.9, type hinting generics have been moved
    # from the "typing" to the "collections" module
    # (see PEP 585: https://peps.python.org/pep-0585/)
    from collections.abc import Callable, Generator, Sequence

    Sliceable = list[Tensor] | tuple[Tensor, ...]

    CountingFunction = Callable[[Tensor, Tensor], Tensor]
    PathLike = str | Path

    Gather = Callable[[Tensor, int, Tensor], Tensor]
    Scatter = Callable[[Tensor, int, Tensor, str], Tensor]
    ScatterOrGather = Gather | Scatter
else:
    from typing import Callable, Generator, List, Sequence, Tuple, Union

    # in Python 3.8, "from __future__ import annotations" only affects
    # type annotations not type aliases
    Sliceable = Union[List[Tensor], Tuple[Tensor, ...]]

    CountingFunction = Callable[[Tensor, Tensor], Tensor]
    PathLike = Union[str, Path]

    Gather = Callable[[Tensor, int, Tensor], Tensor]
    Scatter = Callable[[Tensor, int, Tensor, str], Tensor]
    ScatterOrGather = Union[Gather, Scatter]


class Molecule(TypedDict):
    """Representation of fundamental molecular structure (atom types and postions)."""

    numbers: Tensor
    """Tensor of atomic numbers"""

    positions: Tensor
    """Tensor of 3D coordinates of shape (n, 3)"""


# TODO: Extend with type() and to() methods.
class TensorLike:
    """
    Provide `device` and `dtype` as well as `to()` and `type()` for other
    classes.
    """

    __slots__ = ["__device", "__dtype"]

    def __init__(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        self.__device = (
            device if device is not None else torch.device(defaults.TORCH_DEVICE)
        )
        self.__dtype = dtype if dtype is not None else defaults.TORCH_DTYPE

    @property
    def device(self) -> torch.device:
        """The device on which the class object resides."""
        return self.__device

    @device.setter
    def device(self, *_):
        """
        Instruct users to use the ".to" method if wanting to change device.
        """
        raise AttributeError("Move object to device using the `.to` method")

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by class object."""
        return self.__dtype

    @dtype.setter
    def dtype(self, *_):
        """
        Instruct users to use the `.type` method if wanting to change dtype.
        """
        raise AttributeError("Change object to dtype using the `.type` method")

    def type(self, dtype: torch.dtype) -> Self:
        """
        Returns a copy of the `TensorLike` instance with specified floating
        point type.
        This method creates and returns a new copy of the `TensorLike` instance
        with the specified dtype.

        Parameters
        ----------
        dtype : torch.dtype
            Floating point type.

        Returns
        -------
        TensorLike
            A copy of the `TensorLike` instance with the specified dtype.

        Notes
        -----
        If the `TensorLike` instance has already the desired dtype `self` will
        be returned.
        """
        if self.dtype == dtype:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `type` method requires setting `__slots__` in the "
                f"'{self.__class__.__name__}' class."
            )

        allowed_dtypes = (torch.float16, torch.float32, torch.float64)
        if dtype not in allowed_dtypes:
            raise ValueError(f"Only float types allowed (received '{dtype}').")

        args = {}
        for s in self.__slots__:
            if not s.startswith("__"):
                attr = getattr(self, s)
                if isinstance(attr, Tensor) or issubclass(attr, TensorLike):
                    if attr.dtype in allowed_dtypes:
                        attr = attr.type(dtype)  # type: ignore
                args[s] = attr

        return self.__class__(**args, dtype=dtype)

    def to(self, device: torch.device) -> Self:
        """
        Returns a copy of the `TensorLike` instance on the specified device.

        This method creates and returns a new copy of the `TensorLike` instance
        on the specified device "``device``".

        Parameters
        ----------
        device : torch.device
            Device to which all associated tensors should be moved.

        Returns
        -------
        TensorLike
            A copy of the `TensorLike` instance placed on the specified device.

        Notes
        -----
        If the `TensorLike` instance is already on the desired device `self`
        will be returned.
        """
        if self.device == device:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `to` method requires setting `__slots__` in the "
                f"'{self.__class__.__name__}' class."
            )

        args = {}
        for s in self.__slots__:
            if not s.startswith("__"):
                attr = getattr(self, s)
                if isinstance(attr, Tensor) or issubclass(attr, TensorLike):
                    attr = attr.to(device=device)  # type: ignore
                args[s] = attr

        return self.__class__(**args, device=device)
