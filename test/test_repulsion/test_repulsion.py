"""
Run tests for repulsion contribution.

(Note that the analytical gradient tests fail for `torch.float`.)
"""

from math import sqrt

import pytest
import torch

from dxtb.basis import IndexHelper
from dxtb.classical import Repulsion, new_repulsion
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular, get_elem_param
from dxtb._types import Tensor
from dxtb.utils import batch
from dxtb.utils.exceptions import ParameterWarning

from .samples import samples

sample_list = ["H2O", "SiH4", "MB16_43_01", "MB16_43_02", "LYS_xao"]


def test_none() -> None:
    dummy = torch.tensor(0.0)
    _par = par.copy(deep=True)

    with pytest.warns(ParameterWarning):
        _par.repulsion = None
        assert new_repulsion(dummy, _par) is None

        del _par.repulsion
        assert new_repulsion(dummy, _par) is None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["gfn1"].type(dtype)

    rep = new_repulsion(numbers, par, **dd)
    if rep is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = rep.get_cache(numbers, ihelp)
    e = rep.get_energy(positions, cache)

    assert pytest.approx(ref, abs=tol) == e.sum(-1).item()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample1, sample2 = samples[name1], samples[name2]

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
    ref = torch.stack(
        [
            sample1["gfn1"].type(dtype),
            sample2["gfn1"].type(dtype),
        ],
    )

    rep = new_repulsion(numbers, par, **dd)
    if rep is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = rep.get_cache(numbers, ihelp)
    e = rep.get_energy(positions, cache)

    assert pytest.approx(ref, abs=tol) == e.sum(-1)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["H2O", "SiH4"])
def test_grad_pos_tblite(dtype: torch.dtype, name: str) -> None:
    """Compare with reference values from tblite."""
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype).detach().clone()
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

    assert torch.allclose(ref, grad_backward, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_grad_pos_backward(dtype: torch.dtype, name: str) -> None:
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype).detach().clone()
    positions.requires_grad_(True)

    rep = new_repulsion(numbers, par, **dd)
    if rep is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = rep.get_cache(numbers, ihelp)

    # analytical gradient
    grad_analytical = rep.get_grad(positions, cache)

    # automatic gradient
    energy = torch.sum(rep.get_energy(positions, cache), dim=-1)
    energy.backward()

    if positions.grad is None:
        assert False
    grad_backward = positions.grad.clone()

    assert torch.allclose(grad_analytical, grad_backward, atol=tol)


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
    grad_analytical = rep.get_grad(positions, cache)

    # numerical gradient
    grad_numerical = calc_numerical_gradient(positions, rep, cache)
    assert torch.allclose(grad_analytical, grad_numerical, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_grad_param(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.

    Args
    ----
    dtype : torch.dtype
        Data type of the tensor.
    """
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    if par.repulsion is None:
        assert False

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
        rep = Repulsion(numbers, arep, zeff, kexp, **dd)
        cache = rep.get_cache(numbers, ihelp)
        return rep.get_energy(positions, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, (_arep, _zeff, _kexp), atol=tol)


def calc_numerical_gradient(
    positions: Tensor, rep: Repulsion, cache: "Repulsion.Cache"
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
