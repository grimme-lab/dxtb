"""
Test derivative of (D3) coordination number w.r.t. positions.
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import DD, Callable, CountingFunction, Tensor
from dxtb.ncoord import erf_count, exp_count, get_coordination_number
from dxtb.utils import batch

from ..utils import dgradcheck, dgradgradcheck
from .samples import samples

sample_list = ["SiH4", "PbH4-BiH3", "MB16_43_01"]

tol = 1e-8


def gradchecker(
    dtype: torch.dtype, name: str, cf: CountingFunction
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    # variables to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return get_coordination_number(numbers, pos, cf)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("cf", [exp_count, erf_count])
def test_grad(dtype: torch.dtype, name: str, cf: CountingFunction) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, cf)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("cf", [exp_count, erf_count])
def test_gradgrad(dtype: torch.dtype, name: str, cf: CountingFunction) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, cf)
    assert dgradgradcheck(func, diffvars, atol=tol)


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str, cf: CountingFunction
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    sample1, sample2 = samples[name1], samples[name2]
    numbers = batch.pack(
        [
            sample1["numbers"],
            sample2["numbers"],
        ]
    )
    positions = batch.pack(
        [
            sample1["positions"].type(dtype),
            sample2["positions"].type(dtype),
        ]
    )

    # variables to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return get_coordination_number(numbers, pos, counting_function=cf)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("cf", [exp_count, erf_count])
def test_grad_batch(
    dtype: torch.dtype, name1: str, name2: str, cf: CountingFunction
) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2, cf)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("cf", [exp_count, erf_count])
def test_gradgrad_batch(
    dtype: torch.dtype, name1: str, name2: str, cf: CountingFunction
) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2, cf)
    assert dgradgradcheck(func, diffvars, atol=tol)
