"""
General test for Core Hamiltonian.
"""
from __future__ import annotations

import pytest
import torch

from dxtb.basis import IndexHelper
from dxtb.param import GFN1_XTB as par
from dxtb.xtb import Hamiltonian

from ..utils import get_device_from_str


def test_no_h0_fail() -> None:
    dummy = torch.tensor([])
    _par = par.model_copy(deep=True)
    _par.hamiltonian = None

    with pytest.raises(RuntimeError):
        Hamiltonian(dummy, _par, dummy)  # type: ignore


def test_no_h0_fail2() -> None:
    numbers = torch.tensor([1])
    ihelp = IndexHelper.from_numbers(numbers, {1: [0]})
    _par = par.model_copy(deep=True)
    h0 = Hamiltonian(numbers, _par, ihelp)

    _par.hamiltonian = None
    with pytest.raises(RuntimeError):
        h0._get_hscale()  # pylint: disable=protected-access

    with pytest.raises(RuntimeError):
        h0.build(numbers, numbers)

    with pytest.raises(RuntimeError):
        h0.get_gradient(numbers, numbers, numbers, numbers, numbers, numbers, numbers)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    numbers = torch.tensor([1])
    ihelp = IndexHelper.from_numbers(numbers, {1: [0]})
    h0 = Hamiltonian(numbers, par, ihelp)
    assert h0.type(dtype).dtype == dtype


def test_change_type_fail() -> None:
    numbers = torch.tensor([1])
    ihelp = IndexHelper.from_numbers(numbers, {1: [0]})
    h0 = Hamiltonian(numbers, par, ihelp)

    # trying to use setter
    with pytest.raises(AttributeError):
        h0.dtype = torch.float64

    # passing disallowed dtype
    with pytest.raises(ValueError):
        h0.type(torch.bool)


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    numbers = torch.tensor([1])
    ihelp = IndexHelper.from_numbers(numbers, {1: [0]})
    h0 = Hamiltonian(numbers, par, ihelp)

    device = get_device_from_str(device_str)
    h0 = h0.to(device)
    assert h0.device == device


def test_change_device_fail() -> None:
    numbers = torch.tensor([1])
    ihelp = IndexHelper.from_numbers(numbers, {1: [0]})
    h0 = Hamiltonian(numbers, par, ihelp)

    # trying to use setter
    with pytest.raises(AttributeError):
        h0.device = "cpu"


@pytest.mark.cuda
def test_wrong_device_fail() -> None:
    numbers = torch.tensor([1], device=get_device_from_str("cuda"))
    ihelp = IndexHelper.from_numbers(numbers, {1: [0]})

    # numbers is on a different device
    with pytest.raises(ValueError):
        Hamiltonian(numbers, par, ihelp)
