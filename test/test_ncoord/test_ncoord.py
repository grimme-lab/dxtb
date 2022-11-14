"""
Test calculation of (D3) coordination number.
"""

import pytest
import torch

from dxtb.data import cov_rad_d3
from dxtb.ncoord import (
    exp_count,
    dexp_count,
    erf_count,
    derf_count,
    get_coordination_number,
)
from dxtb.utils import batch
from dxtb.typing import CountingFunction

from .samples import samples

sample_list = ["PbH4-BiH3", "C6H5I-CH3SH"]


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
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    rcov = cov_rad_d3[numbers]
    ref = sample["cn"].type(dtype)

    cn = get_coordination_number(numbers, positions, exp_count, rcov)
    assert pytest.approx(cn) == ref


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_cn_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    sample1, sample2 = samples[name1], samples[name2]
    numbers = batch.pack(
        (
            sample1["numbers"],
            sample2["numbers"],
        )
    )
    positions = batch.pack(
        (
            sample1["positions"].type(dtype),
            sample2["positions"].type(dtype),
        )
    )

    rcov = cov_rad_d3[numbers]
    ref = batch.pack(
        (
            sample1["cn"].type(dtype),
            sample2["cn"].type(dtype),
        )
    )

    cn = get_coordination_number(numbers, positions, exp_count, rcov)
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
    cf, dcf = function

    a = torch.rand(4, dtype=dtype)
    b = torch.rand(4, dtype=dtype)

    a_grad = a.detach().clone().requires_grad_(True)
    count = cf(a_grad, b)

    grad_auto = torch.autograd.grad(count.sum(-1), a_grad)[0]
    grad_expl = dcf(a, b)

    assert pytest.approx(grad_auto) == grad_expl
