"""
Testing the charges module
==========================

The tests surrounding the EEQ charge model include:
 - single molecule
 - batched
 - ghost atoms
 - autograd via `gradcheck`

Note that `torch.linalg.solve` gives slightly different results (around 1e-5
to 1e-6) across different PyTorch versions (1.11.0 vs 1.13.0) for single
precision. For double precision, however the results are identical.
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb import charges
from dxtb._types import DD
from dxtb.utils import batch

from .samples import samples

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["NH3-dimer"])
def test_single(dtype: torch.dtype, name: str):
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    total_charge = sample["total_charge"].to(**dd)
    qref = sample["q"].to(**dd)
    eref = sample["energy"].to(**dd)

    cn = torch.tensor(
        [3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=dtype,
    )
    eeq = charges.ChargeModel.param2019().to(device).type(dtype)
    energy, qat = charges.solve(numbers, positions, total_charge, eeq, cn)

    assert qat.dtype == energy.dtype == dtype
    assert pytest.approx(total_charge, abs=1e-6) == torch.sum(qat, -1)
    assert pytest.approx(qref, abs=tol) == qat
    assert pytest.approx(eref, abs=tol) == energy


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_ghost(dtype: torch.dtype):
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples["NH3-dimer"]
    numbers = sample["numbers"].clone()
    numbers[[1, 5, 6, 7]] = 0
    positions = sample["positions"].to(**dd)
    total_charge = sample["total_charge"].to(**dd)
    cn = torch.tensor(
        [3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        **dd,
    )
    qref = torch.tensor(
        [
            -0.8189238943,
            +0.0000000000,
            +0.2730378155,
            +0.2728482633,
            +0.2730378155,
            +0.0000000000,
            +0.0000000000,
            +0.0000000000,
        ],
        **dd,
    )
    eref = torch.tensor(
        [
            -0.5722096424,
            +0.0000000000,
            +0.1621556977,
            +0.1620431236,
            +0.1621556977,
            +0.0000000000,
            +0.0000000000,
            +0.0000000000,
        ],
        **dd,
    )
    eeq = charges.ChargeModel.param2019(device=device, dtype=dtype)
    energy, qat = charges.solve(numbers, positions, total_charge, eeq, cn)

    assert qat.dtype == energy.dtype == dtype
    assert pytest.approx(total_charge, abs=1e-6) == torch.sum(qat, -1)
    assert pytest.approx(qref, abs=1e-6, rel=tol) == qat
    assert pytest.approx(eref, abs=tol, rel=tol) == energy


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch(dtype: torch.dtype):
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample1, sample2 = (
        samples["PbH4-BiH3"],
        samples["C6H5I-CH3SH"],
    )
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
    total_charge = torch.tensor([0.0, 0.0], **dd)
    eref = batch.pack(
        (
            sample1["energy"].to(**dd),
            sample2["energy"].to(**dd),
        )
    )
    qref = batch.pack(
        (
            sample1["q"].to(**dd),
            sample2["q"].to(**dd),
        )
    )

    cn = torch.tensor(
        [
            [
                3.9195758978,
                0.9835975866,
                0.9835977083,
                0.9835977083,
                0.9832391350,
                2.9579090955,
                0.9874520816,
                0.9874522118,
                0.9874520816,
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
                3.0173754479,
                3.0134898523,
                3.0173773978,
                3.1580192128,
                3.0178688039,
                3.1573804880,
                1.3525004230,
                0.9943449208,
                0.9943846525,
                0.9942776053,
                0.9943862103,
                0.9942779112,
                2.0535643452,
                0.9956985559,
                3.9585744304,
                0.9940553724,
                0.9939077317,
                0.9939362885,
            ],
        ],
        **dd,
    )
    eeq = charges.ChargeModel.param2019(device=device, dtype=dtype)
    energy, qat = charges.solve(numbers, positions, total_charge, eeq, cn)

    assert qat.dtype == energy.dtype == dtype
    assert pytest.approx(total_charge, abs=1e-6) == torch.sum(qat, -1)
    assert pytest.approx(qref, abs=tol) == qat
    assert pytest.approx(eref, abs=tol) == energy
