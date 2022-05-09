import torch
import pytest

from xtbml import charges, utils
from xtbml.exlibs.tbmalt import batch
from xtbml.ncoord.ncoord import erf_count, get_coordination_number


structures = {
    "NH3-dimer": dict(
        numbers=utils.symbol2number("N N H H H H H H".split()),
        positions=torch.tensor(
            [
                [-2.98334550857544, -0.08808205276728, +0.00000000000000],
                [+2.98334550857544, +0.08808205276728, +0.00000000000000],
                [-4.07920360565186, +0.25775116682053, +1.52985656261444],
                [-1.60526800155640, +1.24380481243134, +0.00000000000000],
                [-4.07920360565186, +0.25775116682053, -1.52985656261444],
                [+4.07920360565186, -0.25775116682053, -1.52985656261444],
                [+1.60526800155640, -1.24380481243134, +0.00000000000000],
                [+4.07920360565186, -0.25775116682053, +1.52985656261444],
            ]
        ),
        total_charge=torch.tensor(0.0),
    ),
    "NH3": dict(
        numbers=utils.symbol2number("N H H H".split()),
        positions=torch.tensor(
            [
                [+0.00000000000000, +0.00000000000000, -0.54524837997150],
                [-0.88451840382282, +1.53203081565085, +0.18174945999050],
                [-0.88451840382282, -1.53203081565085, +0.18174945999050],
                [+1.76903680764564, +0.00000000000000, +0.18174945999050],
            ]
        ),
        total_charge=torch.tensor(0.0),
    ),
    "PbH4-BiH3": dict(
        numbers=utils.symbol2number("Pb H H H H Bi H H H".split()),
        positions=torch.tensor(
            [
                [-0.00000020988889, -4.98043478877778, +0.00000000000000],
                [+3.06964045311111, -6.06324400177778, +0.00000000000000],
                [-1.53482054188889, -6.06324400177778, -2.65838526500000],
                [-1.53482054188889, -6.06324400177778, +2.65838526500000],
                [-0.00000020988889, -1.72196703577778, +0.00000000000000],
                [-0.00000020988889, +4.77334244722222, +0.00000000000000],
                [+1.35700257511111, +6.70626379422222, -2.35039772300000],
                [-2.71400388988889, +6.70626379422222, +0.00000000000000],
                [+1.35700257511111, +6.70626379422222, +2.35039772300000],
            ]
        ),
        total_charge=torch.tensor(0.0),
    ),
    "C6H5I-CH3SH": dict(
        numbers=utils.symbol2number("C C C C C C I H H H H H S H C H H H".split()),
        positions=torch.tensor(
            [
                [-1.42754169820131, -1.50508961850828, -1.93430551124333],
                [+1.19860572924150, -1.66299114873979, -2.03189643761298],
                [+2.65876001301880, +0.37736955363609, -1.23426391650599],
                [+1.50963368042358, +2.57230374419743, -0.34128058818180],
                [-1.12092277855371, +2.71045691257517, -0.25246348639234],
                [-2.60071517756218, +0.67879949508239, -1.04550707592673],
                [-2.86169588073340, +5.99660765711210, +1.08394899986031],
                [+2.09930989272956, -3.36144811062374, -2.72237695164263],
                [+2.64405246349916, +4.15317840474646, +0.27856972788526],
                [+4.69864865613751, +0.26922271535391, -1.30274048619151],
                [-4.63786461351839, +0.79856258572808, -0.96906659938432],
                [-2.57447518692275, -3.08132039046931, -2.54875517521577],
                [-5.88211879210329, 11.88491819358157, +2.31866455902233],
                [-8.18022701418703, 10.95619984550779, +1.83940856333092],
                [-5.08172874482867, 12.66714386256482, -0.92419491629867],
                [-3.18311711399702, 13.44626574330220, -0.86977613647871],
                [-5.07177399637298, 10.99164969235585, -2.10739192258756],
                [-6.35955320518616, 14.08073002965080, -1.68204314084441],
            ]
        ),
        total_charge=torch.tensor(0.0),
    ),
}


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

    def func(positions, total_charge):
        return torch.sum(
            charges.solve(numbers, positions, total_charge, eeq, cn)[0], -1
        )

    assert torch.autograd.gradcheck(func, (positions, total_charge))
