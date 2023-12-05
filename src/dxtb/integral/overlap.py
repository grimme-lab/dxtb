"""
Overlap
=======

The GFNn-xTB overlap matrix.
"""
from __future__ import annotations

import torch

from .._types import Any
from ..constants import labels
from .base import BaseIntegral
from .driver.libcint import OverlapLibcint
from .driver.pytorch import OverlapPytorch

__all__ = ["Overlap"]


class Overlap(BaseIntegral):
    """
    Overlap integral from atomic orbitals.
    """

    integral: OverlapLibcint | OverlapPytorch
    """Instance of actual overlap integral type."""

    def __init__(
        self,
        driver: int = labels.INTDRIVER_LIBCINT,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ):
        super().__init__(device=device, dtype=dtype)

        # Determine which overlap class to instantiate based on the type
        if driver == labels.INTDRIVER_LIBCINT:
            self.integral = OverlapLibcint(device=device, dtype=dtype, **kwargs)
        elif driver in (labels.INTDRIVER_PYTORCH, labels.INTDRIVER_PYTORCH2):
            self.integral = OverlapPytorch(device=device, dtype=dtype, **kwargs)
        else:
            raise ValueError(f"Unknown integral driver '{driver}'.")
