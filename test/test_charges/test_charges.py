"""Testing the charges module."""

import torch
import pytest

from xtbml import charges
from xtbml.exlibs.tbmalt import batch
from xtbml.typing import Tensor

from .samples import structures


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_charges_single(dtype: torch.dtype):
    sample = structures["NH3-dimer"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    total_charge = sample["total_charge"].type(dtype)
    cn = torch.tensor(
        [3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=dtype,
    )
    qref = torch.tensor(
        [
            -0.8347351804,
            -0.8347351804,
            +0.2730523336,
            +0.2886305132,
            +0.2730523336,
            +0.2730523336,
            +0.2886305132,
            +0.2730523336,
        ],
        dtype=dtype,
    )
    eref = torch.tensor(
        [
            -0.5832575193,
            -0.5832575193,
            +0.1621643199,
            +0.1714161174,
            +0.1621643199,
            +0.1621643199,
            +0.1714161174,
            +0.1621643199,
        ],
        dtype=dtype,
    )
    eeq = charges.ChargeModel.param2019().type(dtype)

    energy, qat = charges.solve(numbers, positions, total_charge, eeq, cn)
    assert qat.dtype == energy.dtype == dtype
    assert torch.allclose(torch.sum(qat, -1), total_charge, atol=1.0e-7)
    assert torch.allclose(qat, qref)
    assert torch.allclose(energy, eref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_charges_ghost(dtype: torch.dtype):
    sample = structures["NH3-dimer"]
    numbers = sample["numbers"]
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_charges_batch(dtype: torch.dtype):
    sample1, sample2 = (
        structures["PbH4-BiH3"],
        structures["C6H5I-CH3SH"],
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
    eref = torch.tensor(
        [
            [
                +0.1035379745,
                -0.0258195114,
                -0.0258195151,
                -0.0258195151,
                -0.0268938305,
                +0.0422307903,
                -0.0158831963,
                -0.0158831978,
                -0.0158831963,
                +0.0000000000,
                +0.0000000000,
                +0.0000000000,
                +0.0000000000,
                +0.0000000000,
                +0.0000000000,
                +0.0000000000,
                +0.0000000000,
                +0.0000000000,
            ],
            [
                -0.0666956672,
                -0.0649253132,
                -0.0666156432,
                -0.0501240988,
                -0.0004746778,
                -0.0504921903,
                -0.1274747615,
                +0.0665769222,
                +0.0715759533,
                +0.0667190716,
                +0.0711318128,
                +0.0666212167,
                -0.1116992442,
                +0.0720166288,
                -0.1300663998,
                +0.0685131245,
                +0.0679318540,
                +0.0622901437,
            ],
        ],
        dtype=dtype,
    )
    qref = torch.tensor(
        [
            [
                +0.1830965969,
                -0.0434600885,
                -0.0434600949,
                -0.0434600949,
                -0.0452680726,
                +0.0727632554,
                -0.0267371663,
                -0.0267371688,
                -0.0267371663,
                +0.0000000000,
                +0.0000000000,
                +0.0000000000,
                +0.0000000000,
                +0.0000000000,
                +0.0000000000,
                +0.0000000000,
                +0.0000000000,
                +0.0000000000,
            ],
            [
                -0.1029278713,
                -0.1001905841,
                -0.1028043772,
                -0.0774975738,
                -0.0007325498,
                -0.0780660341,
                -0.1962493355,
                +0.1120891066,
                +0.1205055899,
                +0.1123282728,
                +0.1197578368,
                +0.1121635250,
                -0.1711138357,
                +0.1212508178,
                -0.2031014175,
                +0.1153482095,
                +0.1143692362,
                +0.1048709842,
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
def test_charges_grad(dtype: torch.dtype = torch.float64):
    sample = structures["NH3"]
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
