"""
Testing dispersion energy and autodiff.

These tests are taken from https://github.com/awvwgk/tad-dftd3/tree/main/tests
and are only included for the sake of completeness.
"""

import pytest
import tad_dftd3 as d3
import torch

from dxtb.dispersion import DispersionD3, new_dispersion
from dxtb.param import GFN1_XTB as par
from dxtb.typing import Tensor
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
def test_disp_batch(dtype: torch.dtype) -> None:
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
    c6 = batch.pack(
        (
            sample1["c6"].type(dtype),
            sample2["c6"].type(dtype),
        )
    )
    ref = batch.pack(
        (
            sample1["edisp"].type(dtype),
            sample2["edisp"].type(dtype),
        )
    )

    rvdw = d3.data.vdw_rad_d3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
    r4r2 = d3.data.sqrt_z_r4_over_r2[numbers]
    param = dict(a1=0.49484001, s8=0.78981345, a2=5.73083694)

    energy = d3.disp.dispersion(
        numbers, positions, c6, rvdw, r4r2, d3.disp.rational_damping, **param
    )
    assert energy.dtype == dtype
    assert torch.allclose(energy, ref)

    # create copy as par lives in global scope
    _par = par.copy(deep=True)
    if _par.dispersion is None or _par.dispersion.d3 is None:
        assert False

    # set parameters explicitly
    _par.dispersion.d3.a1 = param["a1"]
    _par.dispersion.d3.a2 = param["a2"]
    _par.dispersion.d3.s8 = param["s8"]

    disp = new_dispersion(numbers, _par, **dd)
    if disp is None:
        assert False

    cache = disp.get_cache(numbers)
    edisp = disp.get_energy(positions, cache)
    assert edisp.dtype == dtype
    assert torch.allclose(edisp, ref)
    assert torch.allclose(edisp, energy)


@pytest.mark.grad
def test_grad_pos() -> None:
    dtype = torch.double
    dd = {"dtype": dtype}

    sample = samples["C4H5NCS"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype).detach().clone()

    # variable to be differentiated
    positions.requires_grad_(True)

    disp = new_dispersion(numbers, par, **dd)
    if disp is None:
        assert False

    cache = disp.get_cache(numbers)

    def func(positions: Tensor) -> Tensor:
        return disp.get_energy(positions, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, positions)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
def test_grad_pos_tblite(dtype: torch.dtype) -> None:
    """Compare with reference values from tblite."""
    dd = {"dtype": dtype}

    sample = samples["PbH4-BiH3"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype).detach().clone()
    positions.requires_grad_(True)
    ref = sample["grad"].type(dtype)

    disp = new_dispersion(numbers, par, **dd)
    if disp is None:
        assert False

    cache = disp.get_cache(numbers)

    # automatic gradient
    energy = torch.sum(disp.get_energy(positions, cache), dim=-1)
    energy.backward()

    if positions.grad is None:
        assert False
    grad_backward = positions.grad.clone()

    assert torch.allclose(ref, grad_backward)


@pytest.mark.grad
def test_grad_param() -> None:
    dtype = torch.double
    dd = {"dtype": dtype}

    sample = samples["C4H5NCS"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    param = (
        torch.tensor(1.00000000, requires_grad=True, dtype=dtype),
        torch.tensor(0.78981345, requires_grad=True, dtype=dtype),
        torch.tensor(0.49484001, requires_grad=True, dtype=dtype),
        torch.tensor(5.73083694, requires_grad=True, dtype=dtype),
    )
    label = ("s6", "s8", "a1", "a2")

    def func(*inputs):
        input_param = {label[i]: input for i, input in enumerate(inputs)}
        disp = DispersionD3(numbers, input_param, **dd)
        cache = disp.get_cache(numbers)
        return disp.get_energy(positions, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, param)
