"""
Core Hamiltonian.
"""
from __future__ import annotations

import torch

from .._types import Tensor
from ..basis import IndexHelper
from ..param import Param
from ..xtb.h0_gfn1 import GFN1Hamiltonian
from ..xtb.h0_gfn2 import GFN2Hamiltonian
from .base import BaseIntegral

__all__ = ["Hamiltonian"]


class Hamiltonian(BaseIntegral):
    """
    Hamiltonian integral.
    """

    integral: GFN1Hamiltonian | GFN2Hamiltonian
    """Instance of actual GFN Hamiltonian integral."""

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        ihelp: IndexHelper,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device=device, dtype=dtype)

        if par.meta is not None:
            if par.meta.name is not None:
                if par.meta.name.casefold() == "gfn1-xtb":
                    self.integral = GFN1Hamiltonian(
                        numbers, par, ihelp, device=device, dtype=dtype
                    )
                if par.meta.name.casefold() == "gfn2-xtb":
                    self.integral = GFN2Hamiltonian(
                        numbers, par, ihelp, device=device, dtype=dtype
                    )
