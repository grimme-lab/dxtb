"""
Collection of utility functions for distances calculations.
"""
from __future__ import annotations

import torch

from .._types import Tensor
from .tensors import real_triples

__all__ = ["cdist", "is_linear_molecule", "bond_angle"]


def euclidean_dist_quadratic_expansion(x: Tensor, y: Tensor) -> Tensor:
    """
    Computation of euclidean distance matrix via quadratic expansion (sum of
    squared differences or L2-norm of differences).

    While this is significantly faster than the "direct expansion" or
    "broadcast" approach, it only works for euclidean (p=2) distances.
    Additionally, it has issues with numerical stability (the diagonal slightly
    deviates from zero for ``x=y``). The numerical stability should not pose
    problems, since we must remove zeros anyway for batched calculations.

    For more information, see \
    `this Jupyter notebook <https://github.com/eth-cscs/PythonHPC/blob/master/\
    numpy/03-euclidean-distance-matrix-numpy.ipynb>`__ or \
    `this discussion thread on PyTorch forum <https://discuss.pytorch.org/t/\
    efficient-distance-matrix-computation/9065>`__.

    Parameters
    ----------
    x : Tensor
        First tensor.
    y : Tensor
        Second tensor (with same shape as first tensor).

    Returns
    -------
    Tensor
        Pair-wise distance matrix.
    """
    eps = torch.tensor(
        torch.finfo(x.dtype).eps,
        device=x.device,
        dtype=x.dtype,
    )

    # using einsum is slightly faster than `torch.pow(x, 2).sum(-1)`
    xnorm = torch.einsum("...ij,...ij->...i", x, x)
    ynorm = torch.einsum("...ij,...ij->...i", y, y)

    n = xnorm.unsqueeze(-1) + ynorm.unsqueeze(-2)

    # x @ y.mT
    prod = torch.einsum("...ik,...jk->...ij", x, y)

    # important: remove negative values that give NaN in backward
    return torch.sqrt(torch.clamp(n - 2.0 * prod, min=eps))


def cdist_direct_expansion(x: Tensor, y: Tensor, p: int = 2) -> Tensor:
    """
    Computation of cartesian distance matrix.

    Contrary to `euclidean_dist_quadratic_expansion`, this function allows
    arbitrary powers but is considerably slower.

    Parameters
    ----------
    x : Tensor
        First tensor.
    y : Tensor
        Second tensor (with same shape as first tensor).
    p : int, optional
        Power used in the distance evaluation (p-norm). Defaults to 2.

    Returns
    -------
    Tensor
        Pair-wise distance matrix.
    """
    eps = torch.tensor(
        torch.finfo(x.dtype).eps,
        device=x.device,
        dtype=x.dtype,
    )

    # unsqueeze different dimension to create matrix
    diff = torch.abs(x.unsqueeze(-2) - y.unsqueeze(-3))

    # einsum is nearly twice as fast!
    if p == 2:
        distances = torch.einsum("...ijk,...ijk->...ij", diff, diff)
    else:
        distances = torch.sum(torch.pow(diff, p), -1)

    return torch.pow(torch.clamp(distances, min=eps), 1.0 / p)


def cdist(x: Tensor, y: Tensor | None = None, p: int = 2) -> Tensor:
    """
    Wrapper for cartesian distance computation.

    This currently replaces the use of ``torch.cdist``, which does not handle
    zeros well and produces nan's in the backward pass.

    Additionally, ``torch.cdist`` does not return zero for distances between
    same vectors (see `here
    <https://github.com/pytorch/pytorch/issues/57690>`__).

    Parameters
    ----------
    x : Tensor
        First tensor.
    y : Tensor | None, optional
        Second tensor. If no second tensor is given (default), the first tensor
        is used as the second tensor, too.
    p : int, optional
        Power used in the distance evaluation (p-norm). Defaults to 2.

    Returns
    -------
    Tensor
        Pair-wise distance matrix.
    """
    if y is None:
        y = x

    # faster
    if p == 2:
        return euclidean_dist_quadratic_expansion(x, y)

    return cdist_direct_expansion(x, y, p=p)


################################################


def bond_angle(numbers: Tensor, positions: Tensor) -> Tensor:
    """
    Calculate all bond angles. Also works for batched systems.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers.
    positions : Tensor
        Cartesian coordinates of all atoms in the system (nat, 3).

    Returns
    -------
    Tensor
        Tensor of bond angles.
    """
    # masking utility to avoid NaN's
    zero = torch.tensor(0.0, device=positions.device, dtype=positions.dtype)
    eps = torch.tensor(
        torch.finfo(positions.dtype).eps, device=positions.device, dtype=positions.dtype
    )
    mask = real_triples(numbers, diagonal=True, self=False)

    # Expanding dimensions to compute vectors for all combinations
    p1 = positions.unsqueeze(-2).unsqueeze(-2)  # Shape: [N, 1, 1, 3]
    p2 = positions.unsqueeze(-3).unsqueeze(-2)  # Shape: [1, N, 1, 3]
    p3 = positions.unsqueeze(-3).unsqueeze(-3)  # Shape: [1, 1, N, 3]

    vector1 = p1 - p2  # Shape: [N, N, 1, 3]
    vector2 = p3 - p2  # Shape: [N, N, N, 3]

    # Compute dot product across the last dimension
    dot_product = torch.sum(vector1 * vector2, dim=-1)

    # Compute norms of the vectors
    norm1 = torch.norm(vector1, dim=-1)
    norm2 = torch.norm(vector2, dim=-1)

    # Compute cos(theta) and handle potential numerical issues
    cos_theta = torch.where(mask, dot_product / (norm1 * norm2 + eps), zero)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # Calculate bond angles in degrees
    deg = torch.rad2deg(torch.acos(cos_theta))
    return torch.where(mask, deg, -zero)


def is_linear_molecule(
    numbers: Tensor, positions: Tensor, atol: float = 1e-8, rtol: float = 1e-5
) -> Tensor:
    angles = bond_angle(numbers, positions)

    # mask for values close to 0 or 180 degrees
    close_to_zero = torch.isclose(
        angles, torch.zeros_like(angles), atol=atol, rtol=rtol
    )
    close_to_180 = torch.isclose(
        angles, torch.full_like(angles, 180.0), atol=atol, rtol=rtol
    )

    # combined mask for values that are NOT close to either 0 or 180 degrees
    not_linear_mask = ~(close_to_zero | close_to_180)

    # use summation instead of torch.any() to handle batch dimension
    # only if all the whole mask is False, the molecule is linear
    return not_linear_mask.sum((-1, -2, -3)) == 0
