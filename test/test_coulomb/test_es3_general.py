"""
Run tests for energy contribution from on-site third-order
electrostatic energy (ES3).
"""
from __future__ import annotations

import pytest
import torch

from dxtb.coulomb import thirdorder as es3
from dxtb.param import GFN1_XTB

from ..utils import get_device_from_str


def test_none() -> None:
    dummy = torch.tensor(0.0)
    par = GFN1_XTB.copy(deep=True)

    par.thirdorder = None
    assert es3.new_es3(dummy, par) is None

    del par.thirdorder
    assert es3.new_es3(dummy, par) is None


def test_fail() -> None:
    dummy = torch.tensor(0.0)

    par = GFN1_XTB.copy(deep=True)
    if par.thirdorder is None:
        assert False

    with pytest.raises(NotImplementedError):
        par.thirdorder.shell = True
        es3.new_es3(dummy, par)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    cls = es3.new_es3(torch.tensor(0.0), GFN1_XTB)
    assert cls is not None

    cls = cls.type(dtype)
    assert cls.dtype == dtype


def test_change_type_fail() -> None:
    cls = es3.new_es3(torch.tensor(0.0), GFN1_XTB)
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
    cls = es3.new_es3(torch.tensor(0.0), GFN1_XTB)
    assert cls is not None

    cls = cls.to(device)
    assert cls.device == device


def test_change_device_fail() -> None:
    cls = es3.new_es3(torch.tensor(0.0), GFN1_XTB)
    assert cls is not None

    # trying to use setter
    with pytest.raises(AttributeError):
        cls.device = "cpu"
