from typing import List, TypedDict
from torch import Tensor


class SampleInfo(TypedDict):
    """Format of reference records containing elements and coordinates."""

    elements: List[str]
    """List of element symbols"""

    xyz: Tensor
    """Tensor of 3D coordinates of shape (n, 3)"""
