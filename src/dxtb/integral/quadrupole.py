"""
Quadrupole integral.
"""
from __future__ import annotations

import torch

from .._types import Any, Literal
from .base import BaseIntegral
from .driver.libcint import QuadrupoleLibcint


class Quadrupole(BaseIntegral):
    """
    Quadrupole integral from atomic orbitals.
    """

    integral: QuadrupoleLibcint
    """Instance of actual quadrupole integral type."""

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
            self.integral = QuadrupoleLibcint(device=device, dtype=dtype, **kwargs)
        elif driver == "pytorch":
            raise NotImplementedError(
                "PyTorch versions of multipole moments are not implemented. "
                "Use `libcint` as integral driver."
            )
        else:
            raise ValueError(f"Unknown integral driver '{driver}'.")
