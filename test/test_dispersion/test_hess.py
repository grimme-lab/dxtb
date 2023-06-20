"""
Testing dispersion Hessian (autodiff).
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb.dispersion import new_dispersion
from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch, hessian, reshape_fortran

from .samples import samples

sample_list = ["LiH", "SiH4", "MB16_43_01", "PbH4-BiH3"]


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps)

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = reshape_fortran(
        sample["hessian"].type(dtype),
        torch.Size((numbers.shape[0], 3, numbers.shape[0], 3)),
    )

    # variable to be differentiated
    positions.requires_grad_(True)

    disp = new_dispersion(numbers, par, **dd)
    assert disp is not None

    cache = disp.get_cache(numbers)
    hess = hessian(disp.get_energy, (positions, cache))
    assert pytest.approx(ref, abs=tol, rel=tol) == hess.detach()

    positions.detach_()


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["PbH4-BiH3"])
@pytest.mark.parametrize("name2", sample_list)
def skip_test_batch(dtype: torch.dtype, name1: str, name2) -> None:
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 100

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

    ref = batch.pack(
        [
            reshape_fortran(
                sample1["hessian"].type(dtype),
                torch.Size(
                    (sample1["numbers"].shape[0], 3, sample1["numbers"].shape[0], 3)
                ),
            ),
            reshape_fortran(
                sample2["hessian"].type(dtype),
                torch.Size(
                    (sample2["numbers"].shape[0], 3, sample2["numbers"].shape[0], 3)
                ),
            ),
        ]
    )

    # variable to be differentiated
    positions.requires_grad_(True)

    disp = new_dispersion(numbers, par, **dd)
    assert disp is not None

    cache = disp.get_cache(numbers)
    hess = hessian(disp.get_energy, (positions, cache))
    assert pytest.approx(ref, abs=tol, rel=tol) == hess.detach()

    positions.detach_()
