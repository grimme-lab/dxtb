"""
General halogen bond correction tests
=====================================

Run general tests for halogen bond correction including:
 - invalid parameters
 - change of `dtype` and `device`
"""
import pytest
import torch

from dxtb.classical import new_halogen
from dxtb.param import GFN1_XTB as par

from ..utils import get_device_from_str


def test_none() -> None:
    dummy = torch.tensor(0.0)
    _par = par.model_copy(deep=True)

    _par.halogen = None
    assert new_halogen(dummy, _par) is None

    del _par.halogen
    assert new_halogen(dummy, _par) is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    cls = new_halogen(torch.tensor(0.0), par)
    assert cls is not None

    cls = cls.type(dtype)
    assert cls.dtype == dtype


def test_change_type_fail() -> None:
    cls = new_halogen(torch.tensor(0.0), par)
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
    cls = new_halogen(torch.tensor(0.0), par)
    assert cls is not None

    cls = cls.to(device)
    assert cls.device == device


def test_change_device_fail() -> None:
    cls = new_halogen(torch.tensor(0.0), par)
    assert cls is not None

    # trying to use setter
    with pytest.raises(AttributeError):
        cls.device = "cpu"
