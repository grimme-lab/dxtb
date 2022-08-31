from __future__ import annotations

import torch

from ..typing import Tensor


def maybe_move(x: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
    if x.device != device:
        x = x.to(device)
    if x.dtype != dtype:
        x = x.type(dtype)
    return x


def cdist(positions: Tensor, mask: Tensor) -> Tensor:
    """
    Own implementation of `torch.cdist` to avoid NaN's in backward pass.

    Parameters
    ----------
    positions : Tensor
        Input positions.

    Returns
    -------
    Tensor
        Matripositions of euclidean distances of `positions` with itself.
    """

    # add epsilon to avoid zero division in some terms
    eps = positions.new_tensor(torch.finfo(positions.dtype).eps)

    norm = torch.norm(positions, dim=-1) ** 2
    n = norm.unsqueeze(-1) + norm.unsqueeze(-2)

    # positions @ positions.mT
    prod = torch.einsum("...ik, ...jk -> ...ij", positions, positions)

    # sum of squared differences or L2-norm of differences
    # important: remove negative values that give NaN in backward
    ssd = torch.where(mask, n - 2 * prod, eps)

    return torch.where(mask, torch.sqrt(ssd), eps)
