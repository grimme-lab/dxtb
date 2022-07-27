"""
Type annotations for this project.
"""

from collections.abc import Generator

from typing import (
    Any,
    Callable,
    Dict,
    TypedDict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)
from torch import Tensor

Sliceable = Union[List[Tensor], Tuple[Tensor]]

CountingFunction = Callable[[Tensor, Tensor, Any], Tensor]


class Molecule(TypedDict):
    """Representation of fundamental molecular structure (atom types and postions)."""

    numbers: Tensor
    """Tensor of atomic numbers"""

    positions: Tensor
    """Tensor of 3D coordinates of shape (n, 3)"""
