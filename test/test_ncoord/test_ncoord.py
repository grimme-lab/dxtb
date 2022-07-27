import pytest
import torch

from xtbml.data.radii import cov_rad_d3
from xtbml.exlibs.tbmalt import batch
from xtbml.ncoord import ncoord

from .samples import structures


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cn_single(dtype: torch.dtype):
    sample = structures["PbH4-BiH3"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

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

    cn = ncoord.get_coordination_number(numbers, positions, ncoord.exp_count, rcov)
    assert torch.allclose(cn, ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cn_batch(dtype: torch.dtype):
    sample1, sample2 = structures["PbH4-BiH3"], structures["C6H5I-CH3SH"]
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
    ref = torch.tensor(
        [
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
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
            ],
            [
                3.1393690109,
                3.1313166618,
                3.1393768787,
                3.3153429031,
                3.1376547813,
                3.3148119450,
                1.5363609791,
                1.0035246611,
                1.0122337341,
                1.0036621094,
                1.0121959448,
                1.0036619902,
                2.1570565701,
                0.9981809855,
                3.9841127396,
                1.0146225691,
                1.0123561621,
                1.0085891485,
            ],
        ],
        dtype=dtype,
    )

    cn = ncoord.get_coordination_number(numbers, positions, ncoord.exp_count, rcov)
    assert torch.allclose(cn, ref)
