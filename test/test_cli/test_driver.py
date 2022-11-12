"""
Test command line driver.
"""

from pathlib import Path

import pytest
import torch

from dxtb.cli import argparser, Driver


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_driver(dtype: torch.dtype) -> None:
    file = Path(
        Path(__file__).parent.parent, "test_singlepoint/mols/H2/coord"
    ).resolve()
    ref = torch.tensor(-1.0362714373390, dtype=dtype)

    dtype_str = "float32" if dtype == torch.float else "double"
    opts = f"-v 0 --grad --dtype {dtype_str} {file}"
    args = argparser().parse_args(opts.split())
    d = Driver(args)
    result, _ = d.singlepoint()

    energy = result.total.sum(-1).detach()
    assert pytest.approx(energy) == ref


def test_fail() -> None:
    file = Path(
        Path(__file__).parent.parent, "test_singlepoint/mols/H2/coord"
    ).resolve()

    args = argparser().parse_args([str(file)])

    with pytest.raises(ValueError):
        setattr(args, "method", "xtb")
        Driver(args).singlepoint()

    with pytest.raises(NotImplementedError):
        setattr(args, "method", "gfn2")
        Driver(args).singlepoint()

    with pytest.raises(ValueError):
        setattr(args, "method", "gfn1")
        setattr(args, "guess", "random")
        Driver(args).singlepoint()
