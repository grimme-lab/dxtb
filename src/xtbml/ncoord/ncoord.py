from typing import Optional
from ..typing import CountingFunction, Tensor

from math import pi, sqrt
import torch

from ..constants import KCN, KCN_EEQ
from ..data.covrad import covalent_rad_d3


# TODO: differentiate GFN1 and GFN2
# from xtbml.constants import KCN, KA, KB, R_SHIFT


def get_coordination_number(
    numbers: Tensor,
    positions: Tensor,
    counting_function: CountingFunction,
    rcov: Optional[Tensor] = None,
    cutoff: Optional[Tensor] = None,
    **kwargs,
) -> Tensor:
    """
    Compute fractional coordination number using an exponential counting function.

    Args:
        geometry (Geometry): Molecular structure data
        rcov (Tensor): Covalent radii for each species
        counting_function (Callable): Calculate weight for pair
        kwargs: Pass-through arguments for counting function

    Returns:
        cn (Tensor): Coordination numbers for all atoms
    """

    if cutoff is None:
        cutoff = torch.tensor(25.0, dtype=positions.dtype)
    if rcov is None:
        rcov = covalent_rad_d3[numbers].type(positions.dtype)
    if numbers.shape != rcov.shape:
        raise ValueError(
            "Shape of covalent radii is not consistent with atomic numbers"
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError("Shape of positions is not consistent with atomic numbers")

    eps = torch.tensor(torch.finfo(positions.dtype).eps, dtype=positions.dtype)

    real = numbers != 0
    mask = real.unsqueeze(-2) * real.unsqueeze(-1)
    mask.diagonal(dim1=-2, dim2=-1).fill_(False)

    distances = torch.where(
        mask,
        torch.cdist(positions, positions, p=2),
        eps,
    )

    rc = rcov.unsqueeze(-2) + rcov.unsqueeze(-1)
    cf = torch.where(
        mask * (distances <= cutoff),
        counting_function(distances, rc.type(positions.dtype), **kwargs),
        torch.tensor(0.0, dtype=positions.dtype),
    )
    return torch.sum(cf, dim=-1)


def exp_count(r: Tensor, r0: Tensor, kcn: float = KCN) -> Tensor:
    """
    Exponential counting function for coordination number contributions.

    Args:
        r (Tensor): Current distance.
        r0 (Tensor): Cutoff radius.
        kcn (float, optional): Steepness of the counting function. Defaults to `KCN`.

    Returns:
        Tensor: Count of coordination number contribution.
    """
    return 1.0 / (1.0 + torch.exp(-kcn * (r0 / r - 1.0)))


def dexp_count(r: Tensor, r0: Tensor, kcn: float = KCN) -> Tensor:
    """
    Derivative of the counting function w.r.t. the distance.

    Args:
        r (Tensor): Current distance.
        r0 (Tensor): Cutoff radius.
        kcn (float, optional): Steepness of the counting function. Defaults to `KCN`.

    Returns:
        Tensor: Derivative of count of coordination number contribution.
    """
    expterm = torch.exp(-kcn * (r0 / r - 1.0))
    return (-kcn * r0 * expterm) / (r**2 * ((expterm + 1.0) ** 2))


def erf_count(r: Tensor, r0: Tensor, kcn: float = KCN_EEQ) -> Tensor:
    """
    Error function counting function for coordination number contributions.

    Args:
        r (Tensor): Current distance.
        r0 (Tensor): Cutoff radius.
        kcn (float, optional): Steepness of the counting function. Defaults to `KCN_EEQ`.

    Returns:
        Tensor: Count of coordination number contribution.
    """
    return 0.5 * (1.0 + torch.special.erf(-kcn * (r / r0 - 1.0)))


def derf_count(r: Tensor, r0: Tensor, kcn: float = KCN_EEQ) -> Tensor:
    """
    Derivative of error function counting function w.r.t. the distance.

    Args:
        r (Tensor): Current distance.
        r0 (Tensor): Cutoff radius.
        kcn (float, optional): Steepness of the counting function. Defaults to `KCN_EEQ`.

    Returns:
        Tensor: Count of coordination number contribution.
    """
    return -kcn / sqrt(pi) / r0 * torch.exp(-(kcn**2) * (r - r0) ** 2 / r0**2)
