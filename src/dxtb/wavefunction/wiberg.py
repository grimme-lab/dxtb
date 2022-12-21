"""
Wiberg/Mayer bond orders
========================

Wiberg (or better Mayer) bond orders are calculated from the off-diagonal
elements of the matrix product of the density and the overlap matrix.
"""
from __future__ import annotations

from .._types import Tensor
from ..basis import IndexHelper


def get_bond_order(overlap: Tensor, density: Tensor, ihelp: IndexHelper) -> Tensor:
    """Calculate Wiberg bond orders.

    Parameters
    ----------
    overlap : Tensor
        Overlap matrix.
    density : Tensor
        Density matrix.
    ihelp : IndexHelper
        Helper class for indexing.

    Returns
    -------
    Tensor
        Wiberg bond orders.
    """

    # matrix product PS is not symmetric, since P and S do not commute.
    tmp = density @ overlap

    wbo = ihelp.reduce_orbital_to_atom(tmp * tmp.mT, dim=(-2, -1))
    wbo.diagonal(dim1=-2, dim2=-1).fill_(0.0)

    return wbo
