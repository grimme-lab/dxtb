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

sample_list = ["LiH", "SiH4"]  # "MB16_43_01" requires a lot of RAM

tol = 1e-7


def gradcheck_param(
    dtype: torch.dtype, name: str
) -> tuple[
    Callable[[Tensor, Tensor, Tensor], Tensor],  # autograd function
    tuple[Tensor, Tensor, Tensor],  # differentiable variables
]:
    """Prepare gradient check from `torch.autograd`."""
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    # variables to be differentiated
    _hubbard = get_elem_param(
        torch.unique(numbers),
        par.element,
        "gam",
        pad_val=0,
        **dd,
        requires_grad=True,
    )

    _lhubbard = get_elem_param(
        torch.unique(numbers),
        par.element,
        "lgam",
        pad_val=0,
        **dd,
        requires_grad=True,
    )

    assert par.charge is not None
    _gexp = torch.tensor(par.charge.effective.gexp, **dd, requires_grad=True)

    def func(hubbard: Tensor, lhubbard: Tensor, gexp: Tensor) -> Tensor:
        es2 = ES2(hubbard, lhubbard, gexp=gexp, shell_resolved=True, **dd)
        return es2.get_shell_coulomb_matrix(numbers, positions, ihelp)

    return func, (_hubbard, _lhubbard, _gexp)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_grad_param(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradcheck_param(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01"])
def test_grad_param_large(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradcheck_param(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradgrad_param(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradcheck_param(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01"])
def test_gradgrad_param_large(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradcheck_param(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol)


def gradcheck_param_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> tuple[
    Callable[[Tensor, Tensor, Tensor], Tensor],  # autograd function
    tuple[Tensor, Tensor, Tensor],  # differentiable variables
]:
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

    # variables to be differentiated
    _hubbard = get_elem_param(
        torch.unique(numbers),
        par.element,
        "gam",
        pad_val=0,
        **dd,
        requires_grad=True,
    )

    _lhubbard = get_elem_param(
        torch.unique(numbers),
        par.element,
        "lgam",
        pad_val=0,
        **dd,
        requires_grad=True,
    )

    assert par.charge is not None
    _gexp = torch.tensor(par.charge.effective.gexp, **dd, requires_grad=True)

    def func(hubbard: Tensor, lhubbard: Tensor, gexp: Tensor) -> Tensor:
        es2 = ES2(hubbard, lhubbard, gexp=gexp, shell_resolved=True, **dd)
        return es2.get_shell_coulomb_matrix(numbers, positions, ihelp)

    return func, (_hubbard, _lhubbard, _gexp)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_grad_param_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradcheck_param_batch(dtype, name1, name2)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradgrad_param_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradcheck_param_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol)
