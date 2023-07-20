"""
Run tests for repulsion contribution.

(Note that the analytical gradient tests fail for `torch.float`.)
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb._types import DD, Tensor
from dxtb.basis import IndexHelper
from dxtb.classical import new_repulsion
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import batch, hessian, jac

from ..utils import reshape_fortran
from .samples import samples

sample_list = ["LiH", "SiH4", "MB16_43_01"]

device = None


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 100

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref_hess = reshape_fortran(
        sample["gfn1_hess"].to(**dd),
        torch.Size((numbers.shape[0], 3, numbers.shape[0], 3)),
    )

    # variable to be differentiated
    positions.requires_grad_(True)

    rep = new_repulsion(numbers, par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = rep.get_cache(numbers, ihelp)

    hess = hessian(rep.get_energy, (positions, cache))
    positions.detach_()
    assert pytest.approx(ref_hess, abs=tol, rel=tol) == hess.detach()


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def skip_test_single_alt(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 100

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    rep = new_repulsion(numbers, par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = rep.get_cache(numbers, ihelp)

    positions.requires_grad_(True)

    ref_jac = sample["gfn1_grad"].to(**dd)
    ref_hess = reshape_fortran(
        sample["gfn1_hess"].to(**dd),
        torch.Size((numbers.shape[0], 3, numbers.shape[0], 3)),
    )

    # gradient
    fjac = jac(rep.get_energy, argnums=0)
    jacobian: Tensor = fjac(positions, cache).sum(0)  # type: ignore
    assert pytest.approx(ref_jac, abs=tol, rel=tol) == jacobian.detach()

    # hessian
    def _hessian(f, argnums):
        return jac(jac(f, argnums=argnums), argnums=argnums)

    fhess = _hessian(rep.get_energy, argnums=0)
    hess: Tensor = fhess(positions, cache).sum(0)  # type: ignore
    positions.detach_()
    assert pytest.approx(ref_hess, abs=tol, rel=tol) == hess.detach()


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def skip_test_batch(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 100

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

    ref_hess = batch.pack(
        [
            reshape_fortran(
                sample1["gfn1_hess"].to(**dd),
                torch.Size(
                    (
                        sample1["numbers"].to(device).shape[0],
                        3,
                        sample1["numbers"].to(device).shape[0],
                        3,
                    )
                ),
            ),
            reshape_fortran(
                sample2["gfn1_hess"].to(**dd),
                torch.Size(
                    (sample2["numbers"].shape[0], 3, sample2["numbers"].shape[0], 3)
                ),
            ),
        ]
    )

    rep = new_repulsion(numbers, par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = rep.get_cache(numbers, ihelp)

    positions.requires_grad_(True)

    hess = hessian(rep.get_energy, (positions, cache), is_batched=True)
    positions.detach_()
    assert pytest.approx(ref_hess, abs=tol, rel=tol) == hess.detach()
