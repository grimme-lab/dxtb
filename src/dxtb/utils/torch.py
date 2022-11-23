"""
Collection of utility functions for matrices/tensors.
"""

import torch

from ..typing import Tensor


@torch.jit.script
def real_atoms(numbers: Tensor) -> Tensor:
    return numbers != 0


@torch.jit.script
def real_pairs(numbers: Tensor, diagonal: bool = False) -> Tensor:
    real = real_atoms(numbers)
    mask = real.unsqueeze(-2) * real.unsqueeze(-1)
    if diagonal is False:
        mask *= ~torch.diag_embed(torch.ones_like(real))
    return mask


def t2int(x: Tensor) -> int:
    """
    Convert tensor to int.

    Parameters
    ----------
    x : Tensor
        Tensor to convert.

    Returns
    -------
    int
        Integer value of the tensor.
    """
    return int(x.item())
