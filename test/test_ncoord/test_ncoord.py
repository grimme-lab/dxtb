"""
Test calculation of (D3) coordination number.
"""

import pytest
import torch

from dxtb.data import cov_rad_d3
from dxtb.ncoord import exp_count, get_coordination_number
from dxtb.utils import batch

from .samples import samples

sample_list = ["PbH4-BiH3", "C6H5I-CH3SH"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    rcov = cov_rad_d3[numbers]
    ref = sample["cn"].type(dtype)

    cn = get_coordination_number(numbers, positions, exp_count, rcov)
    assert torch.allclose(cn, ref)


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
    assert torch.allclose(cn, ref)


@pytest.mark.grad
@pytest.mark.parametrize("name", ["PbH4-BiH3"])
def test_grad(name: str):
    dtype = torch.double

    sample = structures[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    positions.requires_grad_(True)

    rcov = cov_rad_d3[numbers]
    ref = torch.tensor(
        [
            3.9388208389,
            0.9832025766,
            0.9832026958,
            0.9832026958,
            0.9865897894,
            2.9714603424,
            0.9870455265,
            0.9870456457,
            0.9870455265,
        ],
        dtype=dtype,
    )

    torch.autograd.set_detect_anomaly(True)

    def func(pos: Tensor) -> Tensor:
        return ncoord.get_coordination_number(numbers, pos, ncoord.exp_count, rcov)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, positions)
