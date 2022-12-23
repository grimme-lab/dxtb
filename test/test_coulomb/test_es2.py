"""
Run generic tests for energy contribution from isotropic second-order
electrostatic energy (ES2).
"""
from __future__ import annotations

import pytest
import torch

from dxtb.coulomb import secondorder as es2
from dxtb.param import GFN1_XTB

from ..utils import get_device_from_str


def test_none() -> None:
    dummy = torch.tensor(0.0)
    par = GFN1_XTB.copy(deep=True)

    par.charge = None
    assert es2.new_es2(dummy, par) is None

    del par.charge
    assert es2.new_es2(dummy, par) is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    cls = es2.new_es2(torch.tensor(0.0), GFN1_XTB)
    assert cls is not None

    cls = cls.type(dtype)
    assert cls.dtype == dtype


def test_change_type_fail() -> None:
    cls = es2.new_es2(torch.tensor(0.0), GFN1_XTB)
    assert cls is not None

    # trying to use setter
    with pytest.raises(AttributeError):
        cls.dtype = torch.float64

    # passing disallowed dtype
    with pytest.raises(ValueError):
        cls.type(torch.bool)


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    device = get_device_from_str(device_str)
    cls = es2.new_es2(torch.tensor(0.0), GFN1_XTB)
    assert cls is not None

    cls = cls.to(device)
    assert cls.device == device


def test_change_device_fail() -> None:
    cls = es2.new_es2(torch.tensor(0.0), GFN1_XTB)
    assert cls is not None

    # trying to use setter
    with pytest.raises(AttributeError):
        cls.device = "cpu"
