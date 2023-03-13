"""
Testing dispersion energy and autodiff.

These tests are taken from https://github.com/awvwgk/tad-dftd3/tree/main/tests
and are only included for the sake of completeness.
"""
from __future__ import annotations

import pytest
import tad_dftd4 as d4
import torch

from dxtb._types import Tensor
from dxtb.dispersion import DispersionD4, new_dispersion
from dxtb.param.gfn2 import GFN2_XTB as par
from dxtb.utils import ParameterWarning, batch

from .samples import samples


def test_none() -> None:
    dummy = torch.tensor(0.0)
    _par = par.copy(deep=True)

    with pytest.warns(ParameterWarning):
        _par.dispersion = None
        assert new_dispersion(dummy, _par) is None

        del _par.dispersion
        assert new_dispersion(dummy, _par) is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch(dtype: torch.dtype) -> None:
    dd = {"dtype": dtype}

    sample1, sample2 = (
        samples["PbH4-BiH3"],
        samples["C6H5I-CH3SH"],
    )
    numbers = batch.pack(
        (
            sample1["numbers"],
            sample2["numbers"],
        )
    )
    positions = batch.pack(
        (
            sample1["positions"].type(dtype),
            sample2["positions"].type(dtype),
        )
    )
    charge = positions.new_zeros(numbers.shape[0])

    param = {
        "a1": positions.new_tensor(0.49484001),
        "a2": positions.new_tensor(5.73083694),
        "s8": positions.new_tensor(0.78981345),
    }

    energy = d4.dftd4(numbers, positions, charge, param)
    assert energy.dtype == dtype

    # create copy as `par` lives in global scope
    _par = par.copy(deep=True)
    if _par.dispersion is None or _par.dispersion.d4 is None:
        assert False

    # set parameters explicitly
    _par.dispersion.d4.a1 = param["a1"]
    _par.dispersion.d4.a2 = param["a2"]
    _par.dispersion.d4.s8 = param["s8"]

    disp = new_dispersion(numbers, _par, charge=charge, **dd)
    if disp is None:
        assert False

    cache = disp.get_cache(numbers)
    edisp = disp.get_energy(positions, cache)
    assert edisp.dtype == dtype
    assert pytest.approx(edisp) == energy


@pytest.mark.grad
def test_grad_pos() -> None:
    dtype = torch.double
    dd = {"dtype": dtype}

    sample = samples["C4H5NCS"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype).detach()
    charge = positions.new_tensor(0.0)

    # variable to be differentiated
    positions.requires_grad_(True)

    disp = new_dispersion(numbers, par, charge, **dd)
    if disp is None:
        assert False

    cache = disp.get_cache(numbers)

    def func(positions: Tensor) -> Tensor:
        return disp.get_energy(positions, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, positions)


@pytest.mark.grad
def test_grad_param() -> None:
    dtype = torch.double
    dd = {"dtype": dtype}

    sample = samples["C4H5NCS"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    charge = positions.new_tensor(0.0)
    param = (
        positions.new_tensor(1.00000000, requires_grad=True),
        positions.new_tensor(0.78981345, requires_grad=True),
        positions.new_tensor(0.49484001, requires_grad=True),
        positions.new_tensor(5.73083694, requires_grad=True),
    )
    label = ("s6", "s8", "a1", "a2")

    def func(*inputs):
        input_param = {label[i]: input for i, input in enumerate(inputs)}
        disp = DispersionD4(numbers, input_param, charge, **dd)
        cache = disp.get_cache(numbers)
        return disp.get_energy(positions, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, param)
