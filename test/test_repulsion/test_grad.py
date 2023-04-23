"""
Run tests for repulsion contribution.

(Note that the analytical gradient tests fail for `torch.float`.)
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb._types import Callable, Tensor
from dxtb.basis import IndexHelper
from dxtb.classical import Repulsion, new_repulsion
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular, get_elem_param

from .samples import samples

sample_list = ["H2O", "SiH4", "MB16_43_01", "MB16_43_02", "LYS_xao"]


@pytest.mark.grad
@pytest.mark.parametrize("name", ["H2O"])
def test_grad_fail(name: str) -> None:
    dtype = torch.double
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    xb = new_repulsion(numbers, par, **dd)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)
    energy = xb.get_energy(positions, cache)

    with pytest.raises(RuntimeError):
        xb.get_gradient(energy, positions)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["H2O", "SiH4"])
def test_grad_pos_tblite(dtype: torch.dtype, name: str) -> None:
    """Compare with reference values from tblite."""
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    positions.requires_grad_(True)
    ref = sample["gfn1_grad"].type(dtype)

    rep = new_repulsion(numbers, par, **dd)
    if rep is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = rep.get_cache(numbers, ihelp)

    # automatic gradient
    energy = torch.sum(rep.get_energy(positions, cache), dim=-1)
    energy.backward()

    if positions.grad is None:
        assert False
    grad_backward = positions.grad.clone()

    assert pytest.approx(ref, abs=tol) == grad_backward

    positions.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_grad_pos_autograd(dtype: torch.dtype, name: str) -> None:
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    rep = new_repulsion(numbers, par, **dd)
    if rep is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = rep.get_cache(numbers, ihelp)

    # analytical gradient
    grad_analytical = rep.get_gradient_analytical(positions, cache)

    positions.requires_grad_(True)
    energy = rep.get_energy(positions, cache)
    grad_autograd = rep.get_gradient(energy, positions)

    assert pytest.approx(grad_analytical, abs=tol) == grad_autograd

    positions.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_grad_pos_backward(dtype: torch.dtype, name: str) -> None:
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    rep = new_repulsion(numbers, par, **dd)
    if rep is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = rep.get_cache(numbers, ihelp)

    # analytical gradient
    grad_analytical = rep.get_gradient_analytical(positions, cache)

    # automatic gradient
    positions.requires_grad_(True)
    energy = torch.sum(rep.get_energy(positions, cache), dim=-1)
    energy.backward()

    if positions.grad is None:
        assert False
    grad_backward = positions.grad.clone()

    assert pytest.approx(grad_analytical, abs=tol) == grad_backward

    positions.detach_()


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_grad_pos_analytical(dtype: torch.dtype, name: str) -> None:
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    rep = new_repulsion(numbers, par, **dd)
    if rep is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = rep.get_cache(numbers, ihelp)

    # analytical gradient
    grad_analytical = rep.get_gradient_analytical(positions, cache)

    # numerical gradient
    grad_numerical = calc_numerical_gradient(positions, rep, cache)
    assert pytest.approx(grad_numerical, abs=tol) == grad_analytical


def gradcheck_param(
    dtype: torch.dtype, name: str
) -> tuple[
    Callable[[Tensor, Tensor, Tensor], Tensor],  # autograd function
    tuple[Tensor, Tensor, Tensor],  # differentiable variables
]:
    """Prepare gradient check from `torch.autograd`."""
    assert par.repulsion is not None

    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    # variables to be differentiated
    _arep = get_elem_param(
        torch.unique(numbers),
        par.element,
        "arep",
        pad_val=0,
        **dd,
        requires_grad=True,
    )
    _zeff = get_elem_param(
        torch.unique(numbers),
        par.element,
        "zeff",
        pad_val=0,
        **dd,
        requires_grad=True,
    )
    _kexp = torch.tensor(par.repulsion.effective.kexp, **dd, requires_grad=True)

    def func(arep: Tensor, zeff: Tensor, kexp: Tensor) -> Tensor:
        rep = Repulsion(arep, zeff, kexp, **dd)
        cache = rep.get_cache(numbers, ihelp)
        return rep.get_energy(positions, cache)

    return func, (_arep, _zeff, _kexp)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_grad_param(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    tol = sqrt(torch.finfo(dtype).eps) * 10
    func, diffvars = gradcheck_param(dtype, name)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_gradgrad_param(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    tol = sqrt(torch.finfo(dtype).eps) * 10
    func, diffvars = gradcheck_param(dtype, name)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradgradcheck

    assert gradgradcheck(func, diffvars, atol=tol)


def calc_numerical_gradient(
    positions: Tensor, rep: Repulsion, cache: Repulsion.Cache
) -> Tensor:
    """Calculate gradient numerically for reference."""

    n_atoms = positions.shape[0]

    # setup numerical gradient
    gradient = torch.zeros((n_atoms, 3), dtype=positions.dtype)
    step = 1.0e-6

    for i in range(n_atoms):
        for j in range(3):
            er, el = 0.0, 0.0

            positions[i, j] += step
            er = rep.get_energy(positions, cache)
            er = torch.sum(er, dim=-1)

            positions[i, j] -= 2 * step
            el = rep.get_energy(positions, cache)
            el = torch.sum(el, dim=-1)

            positions[i, j] += step
            gradient[i, j] = 0.5 * (er - el) / step

    return gradient
