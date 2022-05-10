"""
Type annotations for this project.
"""

from typing import Any, Callable, Dict, TypedDict
from torch import Tensor

CountingFunction = Callable[[Tensor, Tensor, Any], Tensor]


class Molecule(TypedDict):
    """Representation of fundamental molecular structure (atom types and postions)."""

    numbers: Tensor
    """Tensor of atomic numbers"""

    positions: Tensor
    """Tensor of 3D coordinates of shape (n, 3)"""


class Record(Molecule, total=False):
    """Format of reference records containing GFN1-xTB and GFN2-xTB reference values."""

    gfn1: Tensor
    """Reference values for GFN1-xTB"""

    gfn2: Tensor
    """Reference values for GFN1-xTB"""

    total_charge: Tensor
    """Reference values for GFN1-xTB"""


Samples = Dict[str, Record]
