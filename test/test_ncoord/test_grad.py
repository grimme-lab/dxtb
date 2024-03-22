"""
Test derivative of (D3) coordination number w.r.t. positions.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.data.radii import COV_D3

from dxtb._types import DD, Tensor
from dxtb.ncoord import (
    dexp_count,
    exp_count,
    get_coordination_number,
    get_coordination_number_gradient,
)
from dxtb.utils import batch, real_pairs

from .samples import samples

sample_list = ["MB16_43_01", "SiH4"]

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 50

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    cutoff = positions.new_tensor(30.0)
    rcov = COV_D3.to(**dd)[numbers]
    ref = sample["dcndr"].to(**dd)
    ref = ref.reshape(numbers.shape[0], numbers.shape[0], 3).transpose(0, 1)

    dcndr = get_coordination_number_gradient(
        numbers, positions, dexp_count, rcov, cutoff
    )
    numdr = calc_numerical_gradient(numbers, positions, rcov)

    # the same atom gets masked in the PyTorch implementation
    mask = real_pairs(numbers, diagonal=True).unsqueeze(-1)
    numdr = torch.where(mask, numdr, numdr.new_tensor(0.0))
    ref = torch.where(mask, ref, ref.new_tensor(0.0))

    assert pytest.approx(dcndr, abs=tol) == numdr
    assert pytest.approx(dcndr, abs=tol) == ref
    assert pytest.approx(numdr, abs=tol) == ref


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    # slightly higher to avoid 10 / 1536 failing
    tol = sqrt(torch.finfo(dtype).eps) * 50

    sample1, sample2 = samples[name1], samples[name2]
    natoms1, natoms2 = (
        sample1["numbers"].to(device).shape[0],
        sample2["numbers"].shape[0],
    )

    numbers = batch.pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    rcov = COV_D3.to(**dd)[numbers]

    positions = batch.pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    ref = batch.pack(
        (
            sample1["dcndr"].to(**dd).reshape(natoms1, natoms1, 3).transpose(0, 1),
            sample2["dcndr"].to(**dd).reshape(natoms2, natoms2, 3).transpose(0, 1),
        ),
    )

    dcndr = get_coordination_number_gradient(numbers, positions, dexp_count, rcov)

    # the same atom gets masked in the PyTorch implementation
    mask = real_pairs(numbers, diagonal=True).unsqueeze(-1)
    ref = torch.where(mask, ref, ref.new_tensor(0.0))

    assert pytest.approx(dcndr, abs=tol) == ref


def calc_numerical_gradient(numbers: Tensor, positions: Tensor, rcov: Tensor) -> Tensor:
    n_atoms = positions.shape[0]
    positions = positions.type(torch.double)
    rcov = rcov.type(torch.double)

    # setup numerical gradient
    gradient = torch.zeros((n_atoms, n_atoms, 3), dtype=positions.dtype)
    step = 1.0e-6

    for i in range(n_atoms):
        for j in range(3):
            positions[i, j] += step
            cnr = get_coordination_number(numbers, positions, exp_count, rcov=rcov)

            positions[i, j] -= 2 * step
            cnl = get_coordination_number(numbers, positions, exp_count, rcov=rcov)

            positions[i, j] += step
            gradient[i, :, j] = 0.5 * (cnr - cnl) / step

    return gradient
