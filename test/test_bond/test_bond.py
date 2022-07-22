import torch
import pytest

from xtbml import bond, utils
from xtbml.exlibs.tbmalt import batch


structures = {
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
        cn=torch.Tensor(
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
        ),
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
        cn=torch.Tensor(
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
        ),
    ),
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_charges_single(dtype: torch.dtype):
    sample = structures["PbH4-BiH3"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    cn = sample["cn"].type(dtype)
    ref = torch.tensor(
        [
            0.6533,
            0.6533,
            0.6533,
            0.6499,
            0.6533,
            0.6533,
            0.6533,
            0.6499,
            0.5760,
            0.5760,
            0.5760,
            0.5760,
            0.5760,
            0.5760,
        ],
        dtype=dtype,
    )

    bond_order = bond.guess_bond_order(numbers, positions, cn)
    assert bond_order.dtype == dtype
    assert torch.allclose(bond_order[bond_order > 0.3], ref, atol=1.0e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_charges_ghost(dtype: torch.dtype):
    sample = structures["PbH4-BiH3"]
    numbers = sample["numbers"]
    numbers[[0, 1, 2, 3, 4]] = 0
    positions = sample["positions"].type(dtype)
    cn = sample["cn"].type(dtype)
    ref = torch.tensor([0.5760, 0.5760, 0.5760, 0.5760, 0.5760, 0.5760], dtype=dtype)

    bond_order = bond.guess_bond_order(numbers, positions, cn)
    assert bond_order.dtype == dtype
    assert torch.allclose(bond_order[bond_order > 0.3], ref, atol=1.0e-3)


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
    cn = batch.pack(
        (
            sample1["cn"].type(dtype),
            sample2["cn"].type(dtype),
        )
    )
    ref = torch.tensor(
        [
            0.5760,
            0.5760,
            0.5760,
            0.5760,
            0.5760,
            0.5760,
            0.4884,
            0.5180,
            0.4006,
            0.4884,
            0.4884,
            0.4012,
            0.4884,
            0.5181,
            0.4006,
            0.5181,
            0.5144,
            0.4453,
            0.5144,
            0.5145,
            0.4531,
            0.5180,
            0.5145,
            0.4453,
            0.4531,
            0.4012,
            0.4453,
            0.4006,
            0.4453,
            0.4006,
            0.6041,
            0.3355,
            0.6041,
            0.3355,
            0.5645,
            0.5673,
            0.5670,
            0.5645,
            0.5673,
            0.5670,
        ],
        dtype=dtype,
    )

    bond_order = bond.guess_bond_order(numbers, positions, cn)
    assert bond_order.dtype == dtype
    assert torch.allclose(bond_order[bond_order > 0.3], ref, atol=1.0e-3)
