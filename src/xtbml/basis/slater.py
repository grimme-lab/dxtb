"""
Expansion coefficients for Slater functions into primitive Gaussian functions
"""

from __future__ import annotations
import math
import numpy as np
import os.path as op
import torch

from ..typing import Tensor


sto_ng = [
    torch.from_numpy(np.load(op.join(op.dirname(__file__), f"sto-{n}g.npy"))).type(
        torch.float64
    )
    for n in range(1, 7)
]


# Two over pi
top = 2.0 / math.pi

dfactorial = torch.tensor([1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0])
"""
Double factorial up to 7!! for normalization of the Gaussian basis functions.

See `OEIS A001147 <https://oeis.org/A001147>`__.
"""


def to_gauss(
    ng: Tensor,
    n: Tensor,
    l: Tensor,
    zeta: Tensor,
    norm: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Expand Slater function in primitive gaussian functions

    Parameters
    ----
    ng : int
        Number of Gaussian functions for the expansion
    n : int
        Principal quantum number of shell
    l : int
        Azimudal quantum number of shell
    zeta : Tensor
        Exponent of Slater function to expand
    norm : bool, optional
        Include normalization in contraction coefficients.
        Defaults to True.

    Returns
    -------
    (Tensor, Tensor):
        Contraction coefficients of primitive gaussians, can contain normalization,
        and exponents of primitive gaussian functions.
    """

    # we have to use a little hack here,
    # if you pass n and l correctly, everything is fine
    # ityp: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
    #    n: 1 2 3 4 5 2 3 4 5  3  4  5  4  5  5  6  6
    #    l: 0 0 0 0 0 1 1 1 1  2  2  2  3  3  4  0  1

    itype = n + torch.tensor([0, 4, 7, 9, 10])[l].to(zeta.device) - 1
    if n == 6 and ng == 6:
        itype = 15 + l

    _coeff, _alpha = sto_ng[ng - 1][:, itype, :]

    alpha = _alpha.type(zeta.dtype).to(zeta.device) * zeta.unsqueeze(-1) ** 2
    coeff = _coeff.type(zeta.dtype).to(zeta.device)

    # normalize the gaussian if requested
    # <φ|φ> = (2i-1)!!(2j-1)!!(2k-1)!!/(4α)^(i+j+k) · sqrt(π/2α)³
    # N² = (4α)^(i+j+k)/((2i-1)!!(2j-1)!!(2k-1)!!)  · sqrt(2α/π)³
    # N = (4α)^((i+j+k)/2) / sqrt((2i-1)!!(2j-1)!!(2k-1)!!) · (2α/π)^(3/4)
    if norm:
        coeff = coeff * (
            (top * alpha) ** 0.75
            * torch.sqrt(4 * alpha) ** l
            / torch.sqrt(dfactorial[l])
        )

    return (alpha, coeff)
