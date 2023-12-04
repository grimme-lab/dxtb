"""
Collection of PyTorch-based integral drivers.
"""
from __future__ import annotations

from .base import BaseIntDriverPytorch
from .impls import OverlapAG, overlap, overlap_gradient

__all__ = ["IntDriverPytorch", "IntDriverPytorchNoAnalytical"]


class IntDriverPytorch(BaseIntDriverPytorch):
    """
    PyTorch-based integral driver.

    The overlap evaluation function implements a custom backward function
    containing the analytical overlap derivative.

    Note
    ----
    Currently, only the overlap integral is implemented.
    """

    def setup_eval_funcs(self) -> None:
        self.eval_ovlp = OverlapAG.apply  # type: ignore
        self.eval_ovlp_grad = overlap_gradient


class IntDriverPytorchNoAnalytical(BaseIntDriverPytorch):
    """
    PyTorch-based integral driver without analytical derivatives.

    Note
    ----
    Currently, only the overlap integral is implemented.
    """

    def setup_eval_funcs(self) -> None:
        self.eval_ovlp = overlap
        self.eval_ovlp_grad = overlap_gradient
