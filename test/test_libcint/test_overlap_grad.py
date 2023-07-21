"""
Test overlap from libcint.
"""
from __future__ import annotations

from math import sqrt

import numpy as np
import pytest
import torch

from dxtb._types import DD
from dxtb.basis import Basis, IndexHelper
from dxtb.integral import libcint as intor
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular

from ..utils import load_from_npz
from .samples import samples

ref_overlap = np.load("test/test_overlap/grad.npz")

sample_list = ["H2", "LiH", "H2O", "SiH4"]

device = None


def explicit(name: str, dd: DD, tol: float) -> None:
    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = load_from_npz(ref_overlap, name, **dd)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    bas = Basis(numbers, par, ihelp, **dd)
    atombases = bas.create_dqc(positions)

    wrapper = intor.LibcintWrapper(atombases, ihelp)
    s = intor.overlap(wrapper)
    norm = torch.pow(s.diagonal(dim1=-1, dim2=-2), -0.5)

    # (3, norb, norb)
    grad = intor.int1e("ipovlp", wrapper)

    # normalize and move xyz dimension to last, which is required for
    # the reduction (only works with extra dimension in last)
    grad = torch.einsum("...xij,...i,...j->...ijx", grad, norm, norm)

    # (norb, norb, 3) -> (nat, norb, 3)
    grad = ihelp.reduce_orbital_to_atom(grad, dim=-3, extra=True)

    # also account for center j and negative because the integral calculates
    # the nabla w.r.t. the spatial coordinate, not the basis central position
    final_grad = -2 * grad.sum(-2)

    assert pytest.approx(ref, abs=tol) == final_grad


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_explicit(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10
    explicit(name, dd, tol)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01"])
def test_explicit_medium(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = 1e-5
    explicit(name, dd, tol)


def autograd(name: str, dd: DD, tol: float) -> None:
    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = load_from_npz(ref_overlap, name, **dd)

    # variable to be differentiated
    positions.requires_grad_(True)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    bas = Basis(numbers, par, ihelp, **dd)
    atombases = bas.create_dqc(positions)

    wrapper = intor.LibcintWrapper(atombases, ihelp)
    s = intor.overlap(wrapper)
    norm = torch.pow(s.diagonal(dim1=-1, dim2=-2), -0.5)
    s = torch.einsum("...ij,...i,...j->...ij", s, norm, norm)

    (g,) = torch.autograd.grad(s.sum(), positions)
    positions.detach_()

    assert pytest.approx(ref, abs=tol) == g


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_autograd(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10
    autograd(name, dd, tol)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01"])
def test_autograd_medium(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    explicit(name, dd, 1e-5)
