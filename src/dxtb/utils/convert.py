"""
Array Conversions
=================

This module contains safe functions for numpy and pytorch interconversion.
"""
from __future__ import annotations

import numpy as np
import torch

from .._types import Tensor

__all__ = ["numpy_to_tensor", "tensor_to_numpy"]


def numpy_to_tensor(
    x: np.ndarray,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.double,
) -> Tensor:
    """
    Convert a numpy array to a PyTorch tensor.

    Parameters
    ----------
    x : np.ndarray
        Array to convert.
    device : torch.device | None, optional
        Device to store the tensor on. If `None` (default), the device is
        inferred from the `field` argument.
    dtype : torch.dtype | None, optional
        Data type of the tensor. Defaults to `torch.double`.

    Returns
    -------
    Tensor
        Converted PyTorch tensor.
    """
    return torch.from_numpy(x).type(dtype).to(device)


def tensor_to_numpy(x: Tensor, dtype: np.dtype = np.dtype(np.float64)) -> np.ndarray:
    """
    Convert a PyTorch tensor to a numpy array.

    Parameters
    ----------
    x : Tensor
        Tensor to convert.
    dtype : np.dtype, optional
        Data type of the array. Defaults to `np.dtype(np.float64)`.

    Returns
    -------
    np.ndarray
        Converted numpy array
    """
    _x: np.ndarray = x.detach().cpu().numpy()
    return _x.astype(dtype)
