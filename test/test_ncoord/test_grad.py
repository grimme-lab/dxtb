"""
Test derivative of (D3) coordination number w.r.t. positions.
"""

from math import sqrt

import pytest
import torch

from dxtb.data import cov_rad_d3
from dxtb.ncoord import (
    dexp_count,
    exp_count,
    get_coordination_number,
    get_coordination_number_gradient,
)
from dxtb.typing import Tensor
from dxtb.utils import batch, real_pairs

from .samples import samples

sample_list = ["MB16_43_01", "SiH4"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 50

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    rcov = cov_rad_d3[numbers].type(dtype)
    ref = sample["dcndr"].type(dtype)
    ref = ref.reshape(numbers.shape[0], numbers.shape[0], 3).transpose(0, 1)

    dcndr = get_coordination_number_gradient(numbers, positions, dexp_count, rcov)
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
    # slightly higher to avoid 10 / 1536 failing
    tol = sqrt(torch.finfo(dtype).eps) * 50

    sample1, sample2 = samples[name1], samples[name2]
    natoms1, natoms2 = sample1["numbers"].shape[0], sample2["numbers"].shape[0]

    numbers = batch.pack(
        (
            sample1["numbers"],
            sample2["numbers"],
        )
    )
    rcov = cov_rad_d3[numbers]

    positions = batch.pack(
        (
            sample1["positions"].type(dtype),
            sample2["positions"].type(dtype),
        )
    )
    ref = batch.pack(
        (
            sample1["dcndr"].type(dtype).reshape(natoms1, natoms1, 3).transpose(0, 1),
            sample2["dcndr"].type(dtype).reshape(natoms2, natoms2, 3).transpose(0, 1),
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
