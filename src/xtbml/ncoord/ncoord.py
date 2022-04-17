import math
from typing import Dict, Callable, Any
import torch

from xtbml.exlibs.tbmalt import Geometry
from xtbml.constants import KCN

# TODO: differentiate GFN1 and GFN2
# from xtbml.constants import KCN, KA, KB, R_SHIFT


Tensor = torch.Tensor


def get_coordination_number(
    geometry: Geometry,
    rcov: Tensor,
    counting_function: Callable[[Tensor, Tensor, Any], Tensor],
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

    numbers = geometry.atomic_numbers
    distances = geometry.distances
    real = numbers != 0
    mask = ~(real.unsqueeze(-2) * real.unsqueeze(-1))
    torch.diagonal(mask, dim1=-2, dim2=-1)[:] = True

    rc = rcov[numbers].unsqueeze(-2) + rcov[numbers].unsqueeze(-1)
    cf = counting_function(distances, rc.type(distances.dtype), **kwargs)
    cf[mask] = 0
    return torch.sum(cf, dim=-1)


def exp_count(r: Tensor, r0: Tensor, kcn: float = KCN) -> Tensor:
    """
    Exponential counting function for coordination number contributions.

    Args:
        r (Tensor): Current distance.
        r0 (Tensor): Cutoff radius.
        kcn (float): Steepness of the counting function.

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
        kcn (float): Steepness of the counting function.

    Returns:
        Tensor: Derivative of count of coordination number contribution.
    """
    expterm = torch.exp(-kcn * (r0 / r - 1.0))
    return (-kcn * r0 * expterm) / (r**2 * ((expterm + 1.0) ** 2))
