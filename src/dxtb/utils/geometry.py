"""
Collection of utility functions for distances calculations.
"""

from __future__ import annotations

import torch
from tad_mctc.storch import cdist

from .._types import Tensor
from .tensors import real_triples

__all__ = ["cdist", "is_linear_molecule", "bond_angle", "mass_center"]


def bond_angle(numbers: Tensor, positions: Tensor) -> Tensor:
    """
    Calculate all bond angles. Also works for batched systems.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system.
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


def mass_center(masses: Tensor, positions: Tensor) -> Tensor:
    """
    Calculate the center of mass from the atomic coordinates and masses.

    Parameters
    ----------
    masses : Tensor
        Atomic masses (nat, ).
    positions : Tensor
        Cartesian coordinates of all atoms in the system (nat, 3).

    Returns
    -------
    Tensor
        Cartesian coordinates of center of mass.
    """
    s = torch.sum(masses, dim=-1)
    return torch.einsum("...z,...zx->...x", masses, positions) / s
