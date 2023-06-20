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
    tol = sqrt(torch.finfo(dtype).eps) * 10
    func, diffvars = gradchecker(dtype, name)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, diffvars, atol=tol)
    diffvars.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradgrad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    tol = sqrt(torch.finfo(dtype).eps) * 10
    func, diffvars = gradchecker(dtype, name)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradgradcheck

    assert gradgradcheck(func, diffvars, atol=tol)
    diffvars.detach_()


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
    tol = sqrt(torch.finfo(dtype).eps) * 10
    func, diffvars = gradchecker_batch(dtype, name1, name2)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, diffvars, atol=tol)
    diffvars.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradgrad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    tol = sqrt(torch.finfo(dtype).eps) * 10
    func, diffvars = gradchecker_batch(dtype, name1, name2)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradgradcheck

    assert gradgradcheck(func, diffvars, atol=tol)
    diffvars.detach_()
