"""
Test numpy and PyTorch interconversion.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dxtb._types import Tensor
from dxtb.utils import convert

from ..utils import get_device_from_str


def test_np_to_torch_float32() -> None:
    """Test if the dtype is retained."""
    arr = np.zeros((10, 10), dtype=np.float32)
    tensor = convert.numpy_to_tensor(arr)

    assert isinstance(tensor, Tensor)
    assert tensor.dtype == torch.float32


def test_np_to_torch_float64() -> None:
    """Test if the dtype is retained."""
    arr = np.zeros((10, 10), dtype=np.float64)
    tensor = convert.numpy_to_tensor(arr)

    assert isinstance(tensor, Tensor)
    assert tensor.dtype == torch.float64


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_np_to_torch_with_dtype(dtype) -> None:
    arr = np.zeros((10, 10))
    tensor = convert.numpy_to_tensor(arr, dtype=dtype)

    assert isinstance(tensor, Tensor)
    assert tensor.dtype == dtype


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_np_to_torch_with_device(device_str: str) -> None:
    device = get_device_from_str(device_str)

    arr = np.zeros((10, 10))
    tensor = convert.numpy_to_tensor(arr, device=device)

    assert isinstance(tensor, Tensor)
    assert tensor.device == device


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_torch_to_np_with_dtype(dtype) -> None:
    tensor = torch.zeros((10, 10))
    arr = convert.tensor_to_numpy(tensor, dtype=dtype)

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == dtype


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_torch_to_np_with_device(device_str: str) -> None:
    device = get_device_from_str(device_str)

    tensor = torch.zeros((10, 10), device=device)
    arr = convert.tensor_to_numpy(tensor)

    assert isinstance(arr, np.ndarray)
