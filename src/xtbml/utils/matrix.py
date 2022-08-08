"""Collection of utility functions for matrices/tensors."""

from __future__ import annotations
import torch

from ..typing import Any, Tensor


def combinations(x: Tensor, r: int = 2) -> Tensor:
    """
    Generate all combinations of matrix elements.

    This is required for the comparision of overlap and Hmailtonian for
    larger systems because these matrices do not coincide with tblite.
    This is possibly due to switched indices, which were introduced in
    the initial Fortran-to-Python port.

    Parameters
    ----------
    x : Tensor
        Matrix to generate combinations from.

    Returns
    -------
    Tensor
        Matrix of combinations (n, 2).
    """
    return torch.combinations(torch.sort(x.flatten())[0], r)


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


def t2int(x:Tensor) -> int:
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