"""
Collection of utility functions for matrices/tensors.
"""

import torch

from ..typing import Any, Tensor


@torch.jit.script
def real_atoms(numbers: Tensor) -> Tensor:
    return numbers != 0


@torch.jit.script
def real_pairs(numbers: Tensor, diagonal: bool = False) -> Tensor:
    real = real_atoms(numbers)
    mask = real.unsqueeze(-2) * real.unsqueeze(-1)
    if diagonal is True:
        mask *= ~torch.diag_embed(torch.ones_like(real))
    return mask


def load_from_npz(npzfile: Any, name: str, dtype: torch.dtype) -> Tensor:
    """Get torch tensor from npz file

    Parameters
    ----------
    npzfile : Any
        Loaded npz file.
    name : str
        Name of the tensor in the npz file.
    dtype : torch.dtype
        Data type of the tensor.
    Returns
    -------
    Tensor   
                Tensor from the npz file.
    """
    name = name.replace("-", "").lower()
    return torch.from_numpy(npzfile[name]).type(dtype)

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
    zero = positions.new_tensor(0.0)
    eps = positions.new_tensor(torch.finfo(positions.dtype).eps)

    # norm = torch.linalg.norm(positions, dim=-1) ** 2
    norm = torch.pow(positions, 2.0).sum(-1)
    n = norm.unsqueeze(-1) + norm.unsqueeze(-2)

    # positions @ positions.mT
    prod = torch.einsum("...ik, ...jk -> ...ij", positions, positions)

    # sum of squared differences or L2-norm of differences
    # important: remove negative values that give NaN in backward
    _ssd = torch.where(mask, n - 2 * prod, zero)

    # remove small negative values, somehow fails for zero instead of eps
    ssd = torch.where(torch.abs(_ssd) > torch.sqrt(eps), _ssd, eps)

    # add epsilon to avoid zero division in later terms
    return torch.where(mask, torch.sqrt(ssd), eps)
    
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
