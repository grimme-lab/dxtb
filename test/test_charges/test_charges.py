"""Testing the charges module."""

import torch
import pytest

from xtbml import charges
from xtbml.typing import Tensor
from xtbml.utils import batch

from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_single(dtype: torch.dtype):
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
    assert torch.allclose(torch.sum(qat, -1), total_charge, atol=1.0e-7)
    assert torch.allclose(qat, qref)
    assert torch.allclose(energy, eref)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_ghost(dtype: torch.dtype):
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
    assert torch.allclose(torch.sum(qat, -1), total_charge, atol=1.0e-7)
    assert torch.allclose(qat, qref)
    assert torch.allclose(energy, eref)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch(dtype: torch.dtype):
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
    assert torch.allclose(torch.sum(qat, -1), total_charge, atol=1.0e-7)
    assert torch.allclose(qat, qref, atol=1.0e-5)
    assert torch.allclose(energy, eref, atol=1.0e-5)


@pytest.mark.grad
def test_charges_grad(dtype: torch.dtype = torch.double):
    sample = samples["NH3"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    total_charge = sample["total_charge"].type(dtype)
    cn = torch.tensor(
        [3.0, 1.0, 1.0, 1.0],
        dtype=dtype,
    )
    eeq = charges.ChargeModel.param2019().type(dtype)

    positions.requires_grad_(True)
    total_charge.requires_grad_(True)

    def func(positions: Tensor, total_charge: Tensor):
        return torch.sum(
            charges.solve(numbers, positions, total_charge, eeq, cn)[0], -1
        )

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, (positions, total_charge))
