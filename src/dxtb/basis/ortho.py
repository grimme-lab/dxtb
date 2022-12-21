"""Gram-Schmidt orthonormalization routines for contracted Gaussian basis functions."""
from __future__ import annotations

import math

import torch

from .._types import Tensor


def orthogonalize(
    alpha: tuple[Tensor, Tensor], coeff: tuple[Tensor, Tensor]
) -> tuple[Tensor, Tensor]:
    """
    Orthogonalize a contracted Gaussian basis function to an existing basis function.
    The second basis function is orthonormalized against the first basis function.

    Parameters
    ----------
    alpha : (Tensor, Tensor)
        Primitive Gaussian exponents for the shell pair.
    coeff : (Tensor, Tensor)
        Contraction coefficients for the shell pair.

    Returns
    -------
    (Tensor, Tensor)
        Primitive Gaussian exponents and contraction coefficients for the
        orthonormalized basis function.
    """

    coeff_i, coeff_j = coeff
    alpha_i, alpha_j = alpha

    coeff_new = coeff_i.new_zeros(coeff_i.shape[-1] + coeff_j.shape[-1])
    alpha_new = alpha_i.new_zeros(alpha_i.shape[-1] + alpha_j.shape[-1])

    # Calculate overlap between basis functions
    overlap = 0.0
    for ai, ci in zip(alpha_i, coeff_i):
        for aj, cj in zip(alpha_j, coeff_j):
            eab = ai + aj
            oab = 1.0 / eab
            kab = torch.sqrt(math.pi * oab) ** 3
            overlap += ci * cj * kab

    # Create new basis function from the pair which is orthogonal to the first basis function
    alpha_new[: alpha_j.shape[-1]], alpha_new[alpha_j.shape[-1] :] = alpha_j, alpha_i
    coeff_new[: coeff_j.shape[-1]], coeff_new[coeff_j.shape[-1] :] = (
        coeff_j,
        -overlap * coeff_i,
    )

    # Normalization of the new basis function might be off, calculate self overlap
    overlap = alpha_i.new_tensor(0.0)
    for ai, ci in zip(alpha_new, coeff_new):
        for aj, cj in zip(alpha_new, coeff_new):
            eab = ai + aj
            oab = 1.0 / eab
            kab = torch.sqrt(math.pi * oab) ** 3
            overlap += ci * cj * kab

    coeff_new /= torch.sqrt(overlap)

    return alpha_new, coeff_new
