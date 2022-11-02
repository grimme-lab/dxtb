"""
Type annotations for this project.
"""

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, TypedDict, TypeGuard, overload

import torch
from torch import Tensor

Sliceable = list[Tensor] | tuple[Tensor]

CountingFunction = Callable[[Tensor, Tensor], Tensor]

PathLike = str | Path


class Molecule(TypedDict):
    """Representation of fundamental molecular structure (atom types and postions)."""

    numbers: Tensor
    """Tensor of atomic numbers"""

    positions: Tensor
    """Tensor of 3D coordinates of shape (n, 3)"""


# TODO: Extend with type() and to() methods.
class TensorLike:
    """
    Provide `device` and `dtype` for other classes.
    """

    __slots__ = ["__device", "__dtype"]

    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.__device = device
        self.__dtype = dtype

    @property
    def device(self) -> torch.device:
        """The device on which the `IndexHelper` object resides."""
        return self.__device

    @device.setter
    def device(self, *args):
        """Instruct users to use the ".to" method if wanting to change device."""
        raise AttributeError("Move object to device using the `.to` method")

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by IndexHelper object."""
        return self.__dtype
