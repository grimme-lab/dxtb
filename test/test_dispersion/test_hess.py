"""
Testing dispersion Hessian (autodiff).
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb._types import DD, Tensor
from dxtb.dispersion import new_dispersion
from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch, hessian

from ..utils import reshape_fortran
from .samples import samples

sample_list = ["LiH", "SiH4", "MB16_43_01", "PbH4-BiH3"]

device = None


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps)

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = reshape_fortran(
        sample["hessian"].to(**dd),
        torch.Size(2 * (numbers.shape[-1], 3)),
    )
    numref = _numhess(numbers, positions)

    # variable to be differentiated
    positions.requires_grad_(True)

    disp = new_dispersion(numbers, par, **dd)
    assert disp is not None

    cache = disp.get_cache(numbers)
    hess = hessian(disp.get_energy, (positions, cache))
    positions.detach_()

    hess = hess.reshape_as(ref)
    assert ref.shape == numref.shape == hess.shape
    assert pytest.approx(ref, abs=tol, rel=tol) == numref
    assert pytest.approx(ref, abs=tol, rel=tol) == hess.detach()


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["PbH4-BiH3"])
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

    ref = batch.pack(
        [
            reshape_fortran(
                sample1["hessian"].to(**dd),
                torch.Size(2 * (sample1["numbers"].to(device).shape[-1], 3)),
            ),
            reshape_fortran(
                sample2["hessian"].to(**dd),
                torch.Size(2 * (sample2["numbers"].shape[0], 3)),
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


def _numhess(numbers: Tensor, positions: Tensor) -> Tensor:
    """Calculate numerical Hessian for reference."""
    dd = {"device": positions.device, "dtype": positions.dtype}

    disp = new_dispersion(numbers, par, **dd)
    assert disp is not None
    cache = disp.get_cache(numbers)

    hess = torch.zeros(*(*positions.shape, *positions.shape), **dd)
    step = 1.0e-4

    def _gradfcn(positions: Tensor) -> Tensor:
        positions.requires_grad_(True)
        energy = disp.get_energy(positions, cache)
        gradient = disp.get_gradient(energy, positions)
        positions.detach_()
        return gradient.detach()

    for i in range(numbers.shape[0]):
        for j in range(3):
            positions[i, j] += step
            gr = _gradfcn(positions)

            positions[i, j] -= 2 * step
            gl = _gradfcn(positions)

            positions[i, j] += step
            hess[:, :, i, j] = 0.5 * (gr - gl) / step

    return hess
