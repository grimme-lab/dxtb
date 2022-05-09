"""
Type annotations for this project.
"""

from typing import Any, Callable, Dict, List, Optional, TypedDict, Union
from torch import Tensor

CountingFunction = Callable[[Tensor, Tensor, Any], Tensor]


class Molecule(TypedDict):
    symbols: List[str]
    """List of element symbols"""

    positions: Tensor
    """Tensor of 3D coordinates of shape (n, 3)"""


class Record(Molecule):
    """Format of reference records containing references, element symbols and coordinates."""

    gfn1: Tensor
    """Reference values for GFN1-xTB"""

    gfn2: Optional[Tensor]
    """Reference values for GFN2-xTB"""


Structures = Dict[str, Record]
