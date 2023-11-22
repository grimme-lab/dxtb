"""
Run tests for overlap of molecules.
References calculated with tblite 0.3.0.
"""
from __future__ import annotations

from math import sqrt

import numpy as np
import pytest
import torch

from dxtb._types import DD
from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch

from ..utils import load_from_npz
from .samples import samples
from .utils import calc_overlap

ref_overlap = np.load("test/test_overlap/overlap.npz")

molecules = ["H2O", "CH4", "SiH4", "PbH4-BiH3"]

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", molecules)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = load_from_npz(ref_overlap, name, dtype)

    s = calc_overlap(numbers, positions, par, uplo="n", dd=dd)

    assert pytest.approx(s, abs=tol) == s.mT
    assert pytest.approx(s, abs=tol) == ref


@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name1", molecules)
@pytest.mark.parametrize("name2", molecules)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Batched version."""
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack((sample1["numbers"].to(device), sample2["numbers"]))
    positions = batch.pack(
        (sample1["positions"].to(**dd), sample2["positions"].to(**dd))
    )
    ref = batch.pack(
        (
            load_from_npz(ref_overlap, name1, dtype),
            load_from_npz(ref_overlap, name2, dtype),
        )
    )

    s = calc_overlap(numbers, positions, par, uplo="n", dd=dd)

    assert pytest.approx(s, abs=tol) == s.mT
    assert pytest.approx(s, abs=tol) == ref
