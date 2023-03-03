"""
Testing dispersion gradient (autodiff).
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import Tensor
from dxtb.dispersion import DispersionD3, new_dispersion
from dxtb.param import GFN1_XTB as par

from .samples import samples


sample_list = ["PbH4-BiH3"]


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_grad_pos_tblite(dtype: torch.dtype, name: str) -> None:
    """Compare with reference values from tblite."""
    dd = {"dtype": dtype}
    tol = 1e-10

    sample = samples[name]
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

    assert pytest.approx(grad_backward, abs=tol) == ref


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
def test_grad_pos_gradcheck(dtype: torch.dtype) -> None:
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
@pytest.mark.parametrize("name", sample_list)
def test_grad_pos_autograd(dtype: torch.dtype, name: str) -> None:
    dd = {"dtype": dtype}
    tol = 1e-10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype).detach().clone()
    positions.requires_grad_(True)
    ref = sample["grad"].type(dtype)

    disp = new_dispersion(numbers, par, **dd)
    if disp is None:
        assert False

    cache = disp.get_cache(numbers)

    energy = disp.get_energy(positions, cache)
    grad_autograd = disp.get_gradient(energy, positions)

    assert pytest.approx(ref, abs=tol) == grad_autograd


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
def test_grad_param(dtype: torch.dtype) -> None:
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
