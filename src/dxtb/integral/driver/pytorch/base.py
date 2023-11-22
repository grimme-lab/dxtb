"""
Base class for PyTorch-based integrals.
"""
from __future__ import annotations

from ...._types import Literal
from ..labels import DRIVER_PYTORCH


class PytorchImplementation:
    """
    Simple label for `PyTorch`-based integral implementations.
    """

    family: Literal["pytorch"] = DRIVER_PYTORCH
    """Label for integral implementation family"""
