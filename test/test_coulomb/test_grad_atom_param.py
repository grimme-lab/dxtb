"""
Run autograd tests for atom-resolved coulomb matrix contribution.
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb._types import Callable, Tensor
from dxtb.basis import IndexHelper
from dxtb.coulomb import ES2
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular, get_elem_param
from dxtb.utils import batch

from .samples import samples

sample_list = ["LiH", "SiH4", "MB16_43_01"]


def gradcheck_param(
    dtype: torch.dtype, name: str
) -> tuple[
    Callable[[Tensor, Tensor], Tensor],  # autograd function
    tuple[Tensor, Tensor],  # differentiable variables
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

    assert par.charge is not None
    _gexp = torch.tensor(par.charge.effective.gexp, **dd, requires_grad=True)

    def func(hubbard: Tensor, gexp: Tensor) -> Tensor:
        es2 = ES2(hubbard, None, gexp=gexp, shell_resolved=False, **dd)
        return es2.get_atom_coulomb_matrix(numbers, positions, ihelp)

    return func, (_hubbard, _gexp)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
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
@pytest.mark.parametrize("name", sample_list)
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


def gradcheck_param_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> tuple[
    Callable[[Tensor, Tensor], Tensor],  # autograd function
    tuple[Tensor, Tensor],  # differentiable variables
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

    assert par.charge is not None
    _gexp = torch.tensor(par.charge.effective.gexp, **dd, requires_grad=True)

    def func(hubbard: Tensor, gexp: Tensor) -> Tensor:
        es2 = ES2(hubbard, None, gexp=gexp, shell_resolved=False, **dd)
        return es2.get_atom_coulomb_matrix(numbers, positions, ihelp)

    return func, (_hubbard, _gexp)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_grad_param_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    tol = sqrt(torch.finfo(dtype).eps) * 10
    func, diffvars = gradcheck_param_batch(dtype, name1, name2)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    diffvars[0].requires_grad_(False)

    assert gradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradgrad_param_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    tol = sqrt(torch.finfo(dtype).eps) * 10
    func, diffvars = gradcheck_param_batch(dtype, name1, name2)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradgradcheck

    diffvars[0].requires_grad_(False)

    assert gradgradcheck(func, diffvars, atol=tol)
