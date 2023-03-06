"""
Testing the charges module
==========================

This module tests the EEQ charge model including:
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
from dxtb.utils import batch

from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_single(dtype: torch.dtype):
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples["NH3-dimer"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    total_charge = sample["total_charge"].type(dtype)
    qref = sample["q"].type(dtype)
    eref = sample["energy"].type(dtype)

    cn = torch.tensor(
        [3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=dtype,
    )
    eeq = charges.ChargeModel.param2019().type(dtype)
    energy, qat = charges.solve(numbers, positions, total_charge, eeq, cn)

    assert qat.dtype == energy.dtype == dtype
    assert pytest.approx(torch.sum(qat, -1), abs=1e-6) == total_charge
    assert pytest.approx(qat, abs=tol) == qref
    assert pytest.approx(energy, abs=tol) == eref


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_ghost(dtype: torch.dtype):
    tol = sqrt(torch.finfo(dtype).eps)

    sample = samples["NH3-dimer"]
    numbers = sample["numbers"].detach().clone()
    numbers[[1, 5, 6, 7]] = 0
    positions = sample["positions"].type(dtype)
    total_charge = sample["total_charge"].type(dtype)
    cn = torch.tensor(
        [3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=dtype,
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
        dtype=dtype,
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
        dtype=dtype,
    )
    eeq = charges.ChargeModel.param2019().type(dtype)
    energy, qat = charges.solve(numbers, positions, total_charge, eeq, cn)

    assert qat.dtype == energy.dtype == dtype
    assert pytest.approx(torch.sum(qat, -1), abs=1e-6) == total_charge
    assert pytest.approx(qat, abs=tol) == qref
    assert pytest.approx(energy, abs=tol) == eref


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch(dtype: torch.dtype):
    tol = sqrt(torch.finfo(dtype).eps)

    sample1, sample2 = (
        samples["PbH4-BiH3"],
        samples["C6H5I-CH3SH"],
    )
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
    total_charge = torch.tensor([0.0, 0.0], dtype=dtype)
    eref = batch.pack(
        (
            sample1["energy"].type(dtype),
            sample2["energy"].type(dtype),
        )
    )
    qref = batch.pack(
        (
            sample1["q"].type(dtype),
            sample2["q"].type(dtype),
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
        dtype=dtype,
    )
    eeq = charges.ChargeModel.param2019().type(dtype)
    energy, qat = charges.solve(numbers, positions, total_charge, eeq, cn)

    assert qat.dtype == energy.dtype == dtype
    assert pytest.approx(torch.sum(qat, -1), abs=1e-6) == total_charge
    assert pytest.approx(qat, abs=tol) == qref
    assert pytest.approx(energy, abs=tol) == eref
