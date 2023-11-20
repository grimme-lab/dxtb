"""
Overlap
=======

The GFNn-xTB overlap matrix.
"""
from __future__ import annotations

import torch

from .._types import Any, Literal
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
        driver: Literal["libcint", "pytorch"] = "libcint",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ):
        super().__init__(device=device, dtype=dtype)

        # Determine which overlap class to instantiate based on the type
        if driver == "libcint":
            self.integral = OverlapLibcint(device=device, dtype=dtype, **kwargs)
        elif driver == "pytorch":
            self.integral = OverlapPytorch(device=device, dtype=dtype, **kwargs)
        else:
            raise ValueError(f"Unknown integral driver '{driver}'.")
