"""
Data classes for basis construction.
"""

from __future__ import annotations

from dataclasses import dataclass

from tad_mctc.typing import Tensor

__all__ = ["AtomCGTOBasis", "CGTOBasis"]


@dataclass
class CGTOBasis:
    angmom: int
    alphas: Tensor  # (nbasis,)
    coeffs: Tensor  # (nbasis,)
    normalized: bool = True

    def wfnormalize_(self) -> CGTOBasis:
        # will always be normalized already in dxtb because we have to also
        # include the orthonormalization of the H2s against the H1s
        return self


@dataclass
class AtomCGTOBasis:
    atomz: int | float | Tensor
    bases: list[CGTOBasis]
    pos: Tensor  # (ndim,)
