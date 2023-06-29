"""
Run autograd tests for atom-resolved coulomb matrix contribution.
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import Callable, Tensor
from dxtb.basis import IndexHelper
from dxtb.coulomb import ES2
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular, get_elem_param
from dxtb.utils import batch

from .samples import samples
from ..utils import dgradcheck, dgradgradcheck

sample_list = ["LiH", "SiH4"]

tol = 1e-7


def gradchecker(
    dtype: torch.dtype, name: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    assert par.repulsion is not None

    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    hubbard = get_elem_param(
        torch.unique(numbers),
        par.element,
        "gam",
        pad_val=0,
        **dd,
    )
    lhubbard = get_elem_param(
        torch.unique(numbers),
        par.element,
        "lgam",
        pad_val=0,
        **dd,
    )

    assert par.charge is not None
    gexp = torch.tensor(par.charge.effective.gexp, **dd)

    # variables to be differentiated
    positions.requires_grad_(True)

    es2 = ES2(hubbard, lhubbard, gexp=gexp, shell_resolved=True, **dd)

    def func(positions: Tensor) -> Tensor:
        return es2.get_shell_coulomb_matrix(numbers, positions, ihelp)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_grad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01"])
def test_grad_large(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol, fast_mode=True)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradgrad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01"])
def test_gradgrad_large(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol, fast_mode=True)


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    assert par.repulsion is not None

    dd = {"dtype": dtype}

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

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    hubbard = get_elem_param(
        torch.unique(numbers),
        par.element,
        "gam",
        pad_val=0,
        **dd,
    )
    lhubbard = get_elem_param(
        torch.unique(numbers),
        par.element,
        "lgam",
        pad_val=0,
        **dd,
    )

    assert par.charge is not None
    gexp = torch.tensor(par.charge.effective.gexp, **dd)

    # variables to be differentiated
    positions.requires_grad_(True)

    es2 = ES2(hubbard, lhubbard, gexp=gexp, shell_resolved=True, **dd)

    def func(positions: Tensor) -> Tensor:
        return es2.get_shell_coulomb_matrix(numbers, positions, ihelp)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
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
@pytest.mark.parametrize("name2", sample_list)
def test_gradgrad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol)
