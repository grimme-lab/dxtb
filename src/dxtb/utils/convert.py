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


numpy_to_torch_dtype_dict = {
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
    np.dtype(np.int8): torch.int8,
    np.dtype(np.int16): torch.int16,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.uint8): torch.uint8,
}
"""Dict of NumPy dtype -> torch dtype (when the correspondence exists)"""

torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}
"""Dict of torch dtype -> NumPy dtype conversion"""


def numpy_to_tensor(
    x: np.ndarray,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
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
    if dtype is None:
        dtype = numpy_to_torch_dtype_dict.get(x.dtype, torch.double)
    assert dtype is not None

    return torch.from_numpy(x).type(dtype).to(device)


def tensor_to_numpy(x: Tensor, dtype: np.dtype | None = None) -> np.ndarray:
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
    if dtype is None:
        dtype = torch_to_numpy_dtype_dict.get(x.dtype, np.dtype(np.float64))
    assert dtype is not None

    _x: np.ndarray = x.detach().cpu().numpy()
    return _x.astype(dtype)
