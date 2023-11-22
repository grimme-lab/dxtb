"""
Test overlap build from integral container.
"""
from __future__ import annotations

import pytest
import torch

from dxtb import integral as ints
from dxtb._types import DD, Tensor
from dxtb.basis import IndexHelper
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import batch

from .samples import samples

PYTORCH_DRIVER = "pytorch"
device = None


def run(numbers: Tensor, positions: Tensor, dd: DD) -> None:
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    i = ints.Integrals(numbers, par, ihelp, driver=PYTORCH_DRIVER, **dd)

    i.setup_driver(positions)
    assert isinstance(i.driver, ints.driver.IntDriverPytorch)

    i.overlap = ints.Overlap(driver=PYTORCH_DRIVER, **dd)
    i.build_overlap(positions)

    o = i.overlap
    assert o is not None
    assert o.matrix is not None


@pytest.mark.parametrize("name", ["H2"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_single(dtype: torch.dtype, name: str):
    """Overlap matrix for monoatomic molecule should be unity."""
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    run(numbers, positions, dd)


@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", ["LiH"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch(dtype: torch.dtype, name1: str, name2: str):
    """Overlap matrix for monoatomic molecule should be unity."""
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = sample1["numbers"].to(device)
    positions = sample2["positions"].to(**dd)

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

    run(numbers, positions, dd)
