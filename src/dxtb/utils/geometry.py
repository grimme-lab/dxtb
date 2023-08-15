"""
Collection of utility functions for distances calculations.
"""
from __future__ import annotations

import torch

from .._types import Tensor

__all__ = ["cdist", "is_linear_molecule"]


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


def bond_angle(atom1: Tensor, atom2: Tensor, atom3: Tensor) -> Tensor:
    # Calculate the bond angle using the coordinates of three atoms
    vector1 = atom1 - atom2
    vector2 = atom3 - atom2
    cos_theta = torch.dot(vector1, vector2) / (
        torch.norm(vector1) * torch.norm(vector2)
    )
    return torch.rad2deg(torch.acos(cos_theta))


def is_linear_molecule(
    positions: Tensor, atol: float = 1e-8, rtol: float = 1e-5
) -> bool:
    # Calculate the number of atoms in the molecule
    num_atoms = positions.shape[-2]

    if num_atoms < 3:
        return True

    # Iterate over the atoms to calculate the bond angles
    for i in range(1, num_atoms - 1):
        atom1 = positions[i - 1]
        atom2 = positions[i]
        atom3 = positions[i + 1]

        angle = bond_angle(atom1, atom2, atom3)

        # Check if the bond angle is not close to 180 degrees
        if not torch.any(
            torch.isclose(
                angle,
                torch.tensor(
                    [0.0, 180.0], device=positions.device, dtype=positions.dtype
                ),
                atol=atol,
                rtol=rtol,
            )
        ):
            return False

    return True