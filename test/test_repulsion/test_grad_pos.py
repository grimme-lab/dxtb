"""
Run tests for nuclear repulsion gradient.
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb._types import DD, Callable, Tensor
from dxtb.basis import IndexHelper
from dxtb.classical import Repulsion, new_repulsion
from dxtb.classical.repulsion import repulsion_energy, repulsion_gradient
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import batch

from ..utils import dgradcheck, dgradgradcheck
from .samples import samples

sample_list = ["H2O", "SiH4", "MB16_43_01", "MB16_43_02", "LYS_xao"]

tol = 1e-7

device = None


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["H2O", "SiH4"])
def test_backward_vs_tblite(dtype: torch.dtype, name: str) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = sample["gfn1_grad"].to(**dd)

    rep = new_repulsion(numbers, par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = rep.get_cache(numbers, ihelp)

    # automatic gradient
    positions.requires_grad_(True)
    energy = torch.sum(rep.get_energy(positions, cache), dim=-1)
    energy.backward()

    assert positions.grad is not None
    grad_backward = positions.grad.clone()

    # also zero out gradients when using `.backward()`
    grad_backward.detach_()
    positions.detach_()
    positions.grad.data.zero_()

    assert pytest.approx(ref, abs=tol) == grad_backward


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2O", "SiH4"])
@pytest.mark.parametrize("name2", ["H2O", "SiH4"])
def test_backward_batch_vs_tblite(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = batch.pack(
        [
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        ]
    )
    positions = batch.pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    ref = batch.pack(
        [
            sample1["gfn1_grad"].to(**dd),
            sample2["gfn1_grad"].to(**dd),
        ]
    )

    rep = new_repulsion(numbers, par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = rep.get_cache(numbers, ihelp)

    # automatic gradient
    positions.requires_grad_(True)
    energy = torch.sum(rep.get_energy(positions, cache))
    energy.backward()

    assert positions.grad is not None
    grad_backward = positions.grad.clone()

    # also zero out gradients when using `.backward()`
    grad_backward.detach_()
    positions.detach_()
    positions.grad.data.zero_()

    assert pytest.approx(ref, abs=tol) == grad_backward


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_grad_pos_backward_vs_analytical(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    rep = new_repulsion(numbers, par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = rep.get_cache(numbers, ihelp)

    # analytical gradient
    e = repulsion_energy(positions, cache.mask, cache.arep, cache.kexp, cache.zeff)
    grad_analytical = repulsion_gradient(
        e, positions, cache.mask, cache.arep, cache.kexp, reduced=True
    )

    # automatic gradient
    positions.requires_grad_(True)
    energy = torch.sum(rep.get_energy(positions, cache), dim=-1)
    energy.backward()

    assert positions.grad is not None
    grad_backward = positions.grad.clone()

    # also zero out gradients when using `.backward()`
    grad_backward.detach_()
    positions.detach_()
    positions.grad.data.zero_()

    assert pytest.approx(grad_analytical, abs=tol) == grad_backward


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


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_grad_pos_analytical_vs_numerical(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    atol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    rep = new_repulsion(numbers, par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = rep.get_cache(numbers, ihelp)

    # analytical gradient
    e = repulsion_energy(positions, cache.mask, cache.arep, cache.kexp, cache.zeff)
    grad_analytical = repulsion_gradient(
        e, positions, cache.mask, cache.arep, cache.kexp, reduced=True
    )

    # numerical gradient
    grad_numerical = calc_numerical_gradient(positions, rep, cache)
    assert pytest.approx(grad_numerical, abs=atol) == grad_analytical


def gradchecker(
    dtype: torch.dtype, name: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    assert par.repulsion is not None

    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    rep = new_repulsion(numbers, par, **dd)
    assert rep is not None
    cache = rep.get_cache(numbers, ihelp)

    # variables to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return rep.get_energy(pos, cache)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_grad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_gradgrad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol)


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = batch.pack(
        [
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        ]
    )
    positions = batch.pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    rep = new_repulsion(numbers, par, **dd)
    assert rep is not None
    cache = rep.get_cache(numbers, ihelp)

    # variables to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return rep.get_energy(pos, cache)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list + ["MB16_43_03"])
def test_grad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list + ["MB16_43_03"])
def test_gradgrad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol)
