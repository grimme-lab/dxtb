from typing import Any, Dict, List, TypedDict
from torch import Tensor


class Reference(TypedDict):
    """Format of reference values in samples."""

    gfn1: Dict[str, Any]

    gfn2: Dict[str, Any]


class Sample(TypedDict):
    """Format of reference records containing references, element symbols and coordinates."""

    ref: Reference
    """Reference values for GFN1-xTB and GFN2-xTB"""

    symbols: List[str]
    """List of element symbols"""

    positions: Tensor
    """Tensor of 3D coordinates of shape (n, 3)"""
