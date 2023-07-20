"""
Test calculation of (D3) coordination number.
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb._types import DD, CountingFunction
from dxtb.data import cov_rad_d3
from dxtb.ncoord import (
    derf_count,
    dexp_count,
    erf_count,
    exp_count,
    get_coordination_number,
)
from dxtb.utils import batch

from .samples import samples

sample_list = ["PbH4-BiH3", "C6H5I-CH3SH", "SiH4"]

device = None


def test_fail() -> None:
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # rcov wrong shape
    with pytest.raises(ValueError):
        rcov = torch.tensor([1.0])
        get_coordination_number(numbers, positions, exp_count, rcov)

    # wrong numbers
    with pytest.raises(ValueError):
        numbers = torch.tensor([1])
        get_coordination_number(numbers, positions, exp_count)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    cutoff = positions.new_tensor(30.0)
    rcov = cov_rad_d3[numbers]
    ref = sample["cn"].to(**dd)

    cn = get_coordination_number(numbers, positions, exp_count, rcov, cutoff)
    assert pytest.approx(cn) == ref


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = batch.pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    positions = batch.pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )

    ref = batch.pack(
        (
            sample1["cn"].to(**dd),
            sample2["cn"].to(**dd),
        )
    )

    cn = get_coordination_number(numbers, positions, exp_count)
    assert pytest.approx(cn) == ref


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize(
    "function",
    [
        (exp_count, dexp_count),
        (erf_count, derf_count),
    ],
)
def test_count_grad(
    dtype: torch.dtype, function: tuple[CountingFunction, CountingFunction]
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    cf, dcf = function

    a = torch.rand(4, dtype=dtype)
    b = torch.rand(4, dtype=dtype)

    a_grad = a.detach().requires_grad_(True)
    count = cf(a_grad, b)

    grad_auto = torch.autograd.grad(count.sum(-1), a_grad)[0]
    grad_expl = dcf(a, b)

    assert pytest.approx(grad_auto, abs=tol) == grad_expl
