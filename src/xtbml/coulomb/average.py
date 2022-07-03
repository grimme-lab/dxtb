"""Averaging functions for hardnesses in GFN1-xTB."""

from __future__ import annotations
from collections.abc import Callable
import torch

from ..typing import Tensor


AveragingFunction = Callable[[Tensor], Tensor]


def harmonic_average(hubbard: Tensor) -> Tensor:
    """Harmonic averaging function for hardnesses in GFN1-xTB.

    Parameters
    ----------
    hubbard : Tensor
        Hubbard parameters of all elements.

    Returns
    -------
    Tensor
        Harmonic average of the Hubbard parameters.
    """

    hubbard1 = 1.0 / (hubbard)
    return 2.0 / (hubbard1.unsqueeze(-1) + hubbard1.unsqueeze(-2))


def arithmetic_average(hubbard: Tensor) -> Tensor:
    """Arithmetic averaging function for hardnesses in GFN1-xTB.

    Parameters
    ----------
    hubbard : Tensor
        Hubbard parameters of all elements.

    Returns
    -------
    Tensor
        Arithmetic average of the Hubbard parameters.
    """

    return 0.5 * (hubbard.unsqueeze(-1) + hubbard.unsqueeze(-2))


def geometric_average(hubbard: Tensor) -> Tensor:
    """Geometric average function for hardnesses in GFN1-xTB.

    Parameters
    ----------
    hubbard : Tensor
        Hubbard parameters of all elements.

    Returns
    -------
    Tensor
        Geometric average of the Hubbard parameters.
    """

    return torch.sqrt(hubbard.unsqueeze(-1) * hubbard.unsqueeze(-2))
