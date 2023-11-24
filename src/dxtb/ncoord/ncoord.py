"""
Calculation of coordination number with various counting functions.
"""
from __future__ import annotations

import torch

from .._types import Any, CountingFunction, Tensor
from ..constants import xtb
from ..data import cov_rad_d3
from ..utils import cdist, real_pairs
from .count import dexp_count, exp_count


def get_coordination_number(
    numbers: Tensor,
    positions: Tensor,
    counting_function: CountingFunction = exp_count,
    rcov: Tensor | None = None,
    cutoff: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """
    Compute fractional coordination number using an exponential counting function.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system.
    positions : Tensor
        Atomic positions of molecular structure.
    counting_function : CountingFunction
        Calculate weight for pairs.
    rcov : Tensor | None, optional
        Covalent radii for each species. Defaults to `None`.
    cutoff : Tensor | None, optional
        Real-space cutoff. Defaults to `None`.
    kwargs : dict[str, Any]
        Pass-through arguments for counting function.

    Returns
    -------
    Tensor
        Coordination numbers for all atoms.

    Raises
    ------
    ValueError
        If shape mismatch between `numbers`, `positions` and `rcov` is detected.
    """
    dd = {"device": positions.device, "dtype": positions.dtype}

    if cutoff is None:
        cutoff = torch.tensor(xtb.NCOORD_DEFAULT_CUTOFF, **dd)
    if rcov is None:
        rcov = cov_rad_d3[numbers].type(positions.dtype).to(positions.device)
    if numbers.shape != rcov.shape:
        raise ValueError(
            f"Shape of covalent radii {rcov.shape} is not consistent with "
            f"({numbers.shape})."
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            f"Shape of positions ({positions.shape[:-1]}) is not consistent "
            f"with atomic numbers ({numbers.shape})."
        )

    mask = real_pairs(numbers, diagonal=True)
    distances = torch.where(
        mask,
        cdist(positions, positions, p=2),
        torch.tensor(torch.finfo(positions.dtype).eps, **dd),
    )

    rc = rcov.unsqueeze(-2) + rcov.unsqueeze(-1)
    cf = torch.where(
        mask * (distances <= cutoff),
        counting_function(distances, rc.type(positions.dtype), **kwargs),
        torch.tensor(0.0, **dd),
    )
    return torch.sum(cf, dim=-1)


def get_coordination_number_gradient(
    numbers: Tensor,
    positions: Tensor,
    dcounting_function: CountingFunction = dexp_count,
    rcov: Tensor | None = None,
    cutoff: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """
    Compute the derivative of the fractional coordination number with respect
    to atomic positions.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system.
    positions : Tensor
        Atomic positions of molecular structure.
    dcounting_function : CountingFunction
        Derivative of the counting function.
    rcov : Tensor | None, optional
        Covalent radii for each species. Defaults to `None`.
    cutoff : Tensor | None, optional
        Real-space cutoff. Defaults to `None`.
    kwargs : dict[str, Any]
        Pass-through arguments for counting function.

    Returns
    -------
    Tensor
        Coordination numbers for all atoms.

    Raises
    ------
    ValueError
        If shape mismatch between `numbers`, `positions` and `rcov` is detected.
    """
    if cutoff is None:
        cutoff = positions.new_tensor(xtb.NCOORD_DEFAULT_CUTOFF)
    if rcov is None:
        rcov = cov_rad_d3[numbers].type(positions.dtype).to(positions.device)
    if numbers.shape != rcov.shape:
        raise ValueError(
            f"Shape of covalent radii {rcov.shape} is not consistent with "
            f"({numbers.shape})."
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            f"Shape of positions ({positions.shape[:-1]}) is not consistent "
            f"with atomic numbers ({numbers.shape})."
        )

    mask = real_pairs(numbers, diagonal=True)
    distances = torch.where(
        mask,
        cdist(positions, positions, p=2),
        positions.new_tensor(torch.finfo(positions.dtype).eps),
    )

    rc = rcov.unsqueeze(-2) + rcov.unsqueeze(-1)
    dcf = torch.where(
        mask * (distances <= cutoff),
        dcounting_function(distances, rc.type(positions.dtype), **kwargs),
        positions.new_tensor(0.0),
    )

    # (n_batch, n_atoms, n_atoms, 3)
    rij = positions.unsqueeze(-2) - positions.unsqueeze(-3)

    # (n_batch, n_atoms, n_atoms, 3)
    dcf = (dcf / distances).unsqueeze(-1) * rij

    return dcf


def get_dcn(dcndr: Tensor, dedcn: Tensor) -> Tensor:
    """
    Calculate complete derivative for coordination number.

    Parameters
    ----------
    dcndr : Tensor
        Derivative of CN with resprect to atomic positions.
        Shape: (batch, natoms, natoms, 3)
    dedcn : Tensor
        Derivative of energy with respect to CN.
        Shape: (batch, natoms, 3)

    Returns
    -------
    Tensor
        Gradient originating from the coordination number.
    """

    # same atom terms added separately (missing due to mask)
    return torch.einsum("...ijx, ...j -> ...ix", dcndr, dedcn) + (
        dcndr.sum(-2) * dedcn.unsqueeze(-1)
    )
