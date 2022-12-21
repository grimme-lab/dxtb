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

# Python 3.10
if sys.version_info >= (3, 10):  # pragma: >=3.10 cover
    from typing import TypeGuard
else:  # pragma: <3.10 cover
    from typing_extensions import TypeGuard

# Python 3.9 + "from __future__ import annotations"
if sys.version_info >= (3, 9):  # pragma: >=3.9 cover
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

else:  # pragma: <3.9 cover
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
    Provide `device` and `dtype` for other classes.
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
    def device(self, *args):
        """Instruct users to use the ".to" method if wanting to change device."""
        raise AttributeError("Move object to device using the `.to` method")

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by class object."""
        return self.__dtype
