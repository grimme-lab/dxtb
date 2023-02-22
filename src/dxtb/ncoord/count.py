"""
Counting functions and their analytical derivatives.
"""
from __future__ import annotations

from math import pi, sqrt

import torch

from .._types import Tensor
from ..constants import xtb


def exp_count(r: Tensor, r0: Tensor, kcn: float = xtb.KCN) -> Tensor:
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


def dexp_count(r: Tensor, r0: Tensor, kcn: float = xtb.KCN) -> Tensor:
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


def erf_count(r: Tensor, r0: Tensor, kcn: float = xtb.KCN_EEQ) -> Tensor:
    """
    Error function counting function for coordination number contributions.

    Args:
        r (Tensor): Current distance.
        r0 (Tensor): Cutoff radius.
        kcn (float, optional): Steepness of the counting function. Defaults to `KCN_EEQ`.

    Returns:
        Tensor: Count of coordination number contribution.
    """
    return 0.5 * (1.0 + torch.erf(-kcn * (r / r0 - 1.0)))


def derf_count(r: Tensor, r0: Tensor, kcn: float = xtb.KCN_EEQ) -> Tensor:
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
