"""
Run tests for overlap gradient.
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch
from torch.autograd.functional import jacobian
from torch.autograd.gradcheck import gradcheck, gradgradcheck

from dxtb._types import Tensor
from dxtb.basis import IndexHelper, slater
from dxtb.integral import Overlap, mmd
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import t2int

from .samples import samples


@pytest.mark.parametrize("dtype", [torch.double])
def test_ss(dtype: torch.dtype):
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    ng = torch.tensor(6)
    n1, n2 = torch.tensor(2), torch.tensor(1)
    l1, l2 = torch.tensor(0), torch.tensor(0)

    alpha1, coeff1 = slater.to_gauss(ng, n1, l1, torch.tensor(1.0, dtype=dtype))
    alpha2, coeff2 = slater.to_gauss(ng, n2, l2, torch.tensor(1.0, dtype=dtype))

    rndm = torch.tensor(
        [0.13695892585203528, 0.47746994997214642, 0.20729096231197164], **dd
    )
    vec = rndm.detach().requires_grad_(True)
    s = mmd.overlap_gto((l1, l2), (alpha1, alpha2), (coeff1, coeff2), vec)

    # autograd
    gradient = torch.autograd.grad(
        s,
        vec,
        grad_outputs=torch.ones_like(s),
    )[0]

    # reference gradient from tblite
    ref = torch.tensor(
        [
            -1.4063021778353382e-002,
            -4.9026890822886887e-002,
            -2.1284755990262944e-002,
        ],
        **dd,
    )

    step = 1e-6
    for i in range(3):
        rndm[i] += step
        sr = mmd.overlap_gto((l1, l2), (alpha1, alpha2), (coeff1, coeff2), rndm)

        rndm[i] -= 2 * step
        sl = mmd.overlap_gto((l1, l2), (alpha1, alpha2), (coeff1, coeff2), rndm)

        rndm[i] += step
        g = 0.5 * (sr - sl) / step

        assert pytest.approx(gradient[i], abs=tol) == g.item()
        assert pytest.approx(gradient[i], abs=tol) == ref[i]


def test_gradcheck_efunction(dtype: torch.dtype = torch.double) -> None:
    xij = torch.tensor(
        [
            [0.0328428932, 0.0555253364, 0.0625081286, 0.0645959303],
            [0.0555253364, 0.1794814467, 0.2809201777, 0.3286595047],
            [0.0625081286, 0.2809201777, 0.6460560560, 0.9701334238],
            [0.0645959303, 0.3286595047, 0.9701334238, 1.9465910196],
        ],
        dtype=dtype,
    )
    rpi = torch.tensor(
        [
            [
                [
                    [-0.0, -0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                ],
                [
                    [-0.0, -0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                ],
                [
                    [-0.0, -0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                ],
            ]
        ],
        dtype=dtype,
    )
    rpj = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            ]
        ],
        dtype=dtype,
    )
    shape = (1, 3, 4, 4, torch.tensor(1), torch.tensor(1))

    assert gradcheck(
        mmd.recursion.EFunction.apply,
        (xij.clone().requires_grad_(True), rpi, rpj, shape),  # type: ignore
    )
    assert gradgradcheck(
        mmd.recursion.EFunction.apply,
        (xij.clone().requires_grad_(True), rpi, rpj, shape),  # type: ignore
    )

    assert gradcheck(
        mmd.recursion.EFunction.apply,
        (
            xij,
            rpi.clone().requires_grad_(True),
            rpj.clone().requires_grad_(True),
            shape,
        ),  # type: ignore
    )
    assert gradgradcheck(
        mmd.recursion.EFunction.apply,
        (
            xij,
            rpi.clone().requires_grad_(True),
            rpj.clone().requires_grad_(True),
            shape,
        ),  # type: ignore
    )


def test_gradcheck_mmd(dtype: torch.dtype = torch.double) -> None:
    angular = (torch.tensor(0), torch.tensor(0))
    alpha = (
        torch.tensor(
            [7.6119966507, 1.3929016590, 0.3869633377, 0.1284296513], dtype=dtype
        ),
        torch.tensor(
            [
                106.3897247314,
                19.5107936859,
                5.4829540253,
                0.7840746045,
                0.3558612764,
                0.1697082371,
            ],
            dtype=dtype,
        ),
    )
    coeff = (
        torch.tensor(
            [0.1853610128, 0.2377167493, 0.1863220334, 0.0445896946], dtype=dtype
        ),
        torch.tensor(
            [
                -0.0980091542,
                -0.1367608607,
                -0.1315236390,
                0.1987189353,
                0.1845818311,
                0.0322807245,
            ],
            dtype=dtype,
        ),
    )
    vec = torch.tensor(
        [
            [-0.0000000000, -0.0000000000, +1.3999999762],
            [-0.0000000000, -0.0000000000, -1.3999999762],
        ],
        dtype=dtype,
    )

    assert gradcheck(
        mmd.recursion.mmd_recursion,
        (angular, alpha, coeff, vec.clone().requires_grad_(True)),  # type: ignore
    )
    assert gradgradcheck(
        mmd.recursion.mmd_recursion,
        (angular, alpha, coeff, vec.clone().requires_grad_(True)),  # type: ignore
    )
    assert gradcheck(
        mmd.explicit.mmd_explicit,
        (angular, alpha, coeff, vec.clone().requires_grad_(True)),  # type: ignore
    )
    assert gradgradcheck(
        mmd.explicit.mmd_explicit,
        (angular, alpha, coeff, vec.clone().requires_grad_(True)),  # type: ignore
    )


@pytest.mark.filterwarnings("ignore")  # works for single precision
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2O", "CH4", "SiH4"])
def test_gradcheck_overlap(dtype: torch.dtype, name: str):
    """Pytorch gradcheck for overlap calculation."""
    dd = {"dtype": dtype}
    tol = 3e-01

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    positions.requires_grad_(True)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    overlap = Overlap(numbers, par, ihelp, **dd)

    def func(pos: Tensor) -> Tensor:
        return overlap.build(pos)

    assert gradcheck(func, (positions), atol=tol)

    positions.detach_()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H", "C", "Rn", "H2O", "CH4", "SiH4"])
def test_overlap_jacobian(dtype: torch.dtype, name: str):
    """Jacobian calculation with AD and numerical gradient."""
    rtol, atol = 1e-1, 3e-1

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    overlap = Overlap(numbers, par, ihelp, **{"dtype": dtype})

    # numerical gradient
    ngrad = calc_numerical_gradient(overlap, positions)  # [natm, norb, norb, 3]

    # autograd jacobian
    positions.requires_grad_(True)
    jac = jacobian(lambda x: overlap.build(x), positions)  # [norb, norb, natm, 3]
    jac = torch.movedim(jac, -2, 0)

    # check whether dimensions are swapped
    assert torch.equal(ngrad - jac, ngrad - torch.movedim(jac, 1, 2))

    # jacobian and numerical gradient mismatch
    # print("diff", ngrad - jac)
    # print("max diff", torch.max(ngrad - jac))

    assert pytest.approx(ngrad, rel=rtol, abs=atol) == jac

    positions.detach_()


def calc_numerical_gradient(overlap: Overlap, positions: Tensor) -> Tensor:
    # setup numerical gradient
    step = 1.0e-4  # require larger deviations for overlap change
    natm = positions.shape[0]
    norb = t2int(overlap.ihelp.orbitals_per_shell.sum())
    gradient = positions.new_zeros((natm, norb, norb, 3))

    # coordinates shift preferred to orbitals shift
    for i in range(natm):
        for j in range(3):
            positions[i, j] += step
            sR = overlap.build(positions)  # [norb, norb]

            positions[i, j] -= 2 * step
            sL = overlap.build(positions)

            positions[i, j] += step
            gradient[i, :, :, j] = 0.5 * (sR - sL) / step

    return gradient


def compare_mmd(
    cgtoi: Tensor,
    cgtoj: Tensor,
    vec: Tensor,
    ovlp_ref: Tensor,
    ovlp_grad_ref: Tensor,
    dtype: torch.dtype,
) -> None:
    """Helper method to compare MMD overlap and gradient with references.
    Parameters
    ----------
    cgtoi : Tensor
        Specification for first CGTO, containing ng, n, l
    cgtoj : Tensor
        Specification for second CGTO, containing ng, n, l
    vec : Tensor
        Shift vector of two CGTO centers
    ovlp_ref : Tensor
        Reference for overlap value
    ovlp_grad_ref : Tensor
        Reference for overlap gradient value
    dtype : torch.dtype
        Dtype for tensors
    """

    # define tolerances
    atol = 1.0e-5

    ngi, ni, li = cgtoi
    ngj, nj, lj = cgtoj
    alpha_i, coeff_i = slater.to_gauss(ngi, ni, li, torch.tensor(1.0, dtype=dtype))
    alpha_j, coeff_j = slater.to_gauss(ngj, nj, lj, torch.tensor(1.0, dtype=dtype))

    # overlap
    ovlp = mmd.overlap_gto((li, lj), (alpha_i, alpha_j), (coeff_i, coeff_j), vec)
    assert pytest.approx(ovlp, abs=atol) == ovlp_ref

    # overlap gradient
    ovlp2, ovlp_grad = mmd.overlap_gto_grad(
        (li, lj), (alpha_i, alpha_j), (coeff_i, coeff_j), vec
    )

    # check overlap from gradient calculation
    assert torch.allclose(ovlp, ovlp2, atol=1e-04)

    # obtain Fortran ordering (row wise)
    ovlp_grad = torch.squeeze(ovlp_grad, 0)
    ovlp_grad = torch.stack([ovlp_grad[i].flatten() for i in range(3)]).transpose(0, 1)
    assert pytest.approx(ovlp_grad, abs=atol) == ovlp_grad_ref


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_overlap_grad_single_1(dtype: torch.dtype):
    """Comparison of single gaussians. Reference values taken from tblite MMD implementation."""

    # define CGTOs (by ng, n, l)
    cgtoi = torch.tensor([6, 2, 0])
    cgtoj = torch.tensor([6, 1, 0])
    vec = -torch.tensor([0.00, 0.00, 1.405], dtype=dtype)

    # define references
    ovlp_ref = torch.tensor([[0.762953]])
    ovlp_grad_ref = torch.tensor([0.000000, 0.000000, -0.145266]).reshape([1, 3])

    compare_mmd(cgtoi, cgtoj, vec, ovlp_ref, ovlp_grad_ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_overlap_grad_single_2(dtype: torch.dtype):
    """Comparison of single gaussians. Reference values taken from tblite MMD implementation."""

    # define CGTOs (by ng, n, l)
    cgtoi = torch.tensor([6, 3, 0])
    cgtoj = torch.tensor([6, 4, 0])
    vec = -torch.tensor([0.00, 0.00, 1.405], dtype=dtype)

    # define references
    ovlp_ref = torch.tensor([[0.893829]])
    ovlp_grad_ref = torch.tensor([0.000000, 0.000000, -0.056111]).reshape([1, 3])

    compare_mmd(cgtoi, cgtoj, vec, ovlp_ref, ovlp_grad_ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_overlap_grad_single_3(dtype: torch.dtype):
    """Comparison of single gaussians. Reference values taken from tblite MMD implementation."""

    # define CGTOs (by ng, n, l)
    cgtoi = torch.tensor([6, 3, 2])
    cgtoj = torch.tensor([6, 4, 2])
    vec = -torch.tensor([6.2571, -3.5633, -2.5369], dtype=dtype)

    # define references
    ovlp_ref = torch.tensor(
        [
            [-0.010617, 0.136801, -0.077907, 0.020332, -0.034273],
            [0.136801, -0.067939, 0.063984, -0.004581, 0.111605],
            [-0.077907, 0.063984, 0.007976, 0.108614, -0.000671],
            [0.020332, -0.004581, 0.108614, -0.140475, -0.137159],
            [-0.034273, 0.111605, -0.000671, -0.137159, 0.009360],
        ]
    )
    ovlp_grad_ref = torch.tensor(
        [
            [0.019315, -0.010999, 0.067906, -0.024156, 0.026208],
            [-0.002824, 0.026208, 0.006939, 0.001608, -0.018706],
            [0.021608, -0.015547, 0.042267, -0.017572, 0.026208],
            [-0.024156, 0.026208, -0.002824, -0.004860, -0.017684],
            [-0.075231, -0.017684, -0.002062, 0.063931, -0.039023],
            [-0.030809, -0.001399, -0.012271, 0.005631, -0.017684],
            [0.026208, 0.006939, 0.001608, -0.017684, -0.002062],
            [0.063931, -0.002062, 0.021626, 0.000621, -0.030809],
            [0.017028, -0.019655, 0.005631, 0.049347, -0.002062],
            [-0.018706, 0.021608, -0.015547, -0.039023, -0.030809],
            [-0.001399, -0.030809, 0.017028, -0.019655, 0.072733],
            [0.046262, -0.028025, -0.010818, -0.041732, -0.030809],
            [0.042267, -0.017572, 0.026208, -0.012271, 0.005631],
            [-0.017684, 0.005631, 0.049347, -0.002062, -0.010818],
            [-0.041732, -0.030809, -0.013359, -0.080075, 0.005631],
        ]
    ).reshape([25, 3])

    compare_mmd(cgtoi, cgtoj, vec, ovlp_ref, ovlp_grad_ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_overlap_grad_single_4(dtype: torch.dtype):
    """Comparison of single gaussians. Reference values taken from tblite MMD implementation."""

    # define CGTOs (by ng, n, l)
    cgtoi = torch.tensor([6, 2, 1])
    cgtoj = torch.tensor([6, 1, 0])
    vec = torch.tensor([1.245, -6.789, 0.123], dtype=dtype)

    # define references
    ovlp_ref = torch.tensor([[-0.081147, 0.001470, 0.014881]]).reshape([3, 1])
    ovlp_grad_ref = torch.tensor(
        [
            [-0.011176, 0.048990, -0.001104],
            [0.000202, -0.001104, -0.011933],
            [-0.009903, -0.011176, 0.000202],
        ]
    )

    compare_mmd(cgtoi, cgtoj, vec, ovlp_ref, ovlp_grad_ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_overlap_grad_single_5(dtype: torch.dtype):
    """Comparison of single gaussians. Reference values taken from tblite MMD implementation."""

    # define CGTOs (by ng, n, l)
    cgtoi = torch.tensor([6, 4, 3])
    cgtoj = torch.tensor([6, 2, 1])
    vec = -torch.tensor([1.245, -6.789, 0.123], dtype=dtype)

    # define references
    ovlp_ref = torch.tensor(
        [
            [-0.006903, -0.149172, 0.001266],
            [-0.028473, 0.002478, -0.055729],
            [0.094314, -0.013510, -0.028473],
            [-0.007895, -0.180163, 0.003929],
            [-0.004323, -0.068378, -0.005745],
            [-0.102090, 0.003369, -0.186877],
            [0.086376, -0.005569, -0.140234],
        ]
    )
    ovlp_grad_ref = torch.tensor(
        [
            [0.000405, -0.001190, -0.055980],
            [0.010266, -0.055980, 0.006905],
            [0.000943, 0.000405, 0.010266],
            [-0.021205, -0.004884, 0.001460],
            [0.001722, 0.001460, 0.020093],
            [0.012277, -0.021205, 0.001722],
            [-0.004884, -0.019109, -0.005973],
            [0.001460, -0.005973, -0.109565],
            [-0.021205, -0.004884, 0.001460],
            [0.002021, -0.000921, -0.064069],
            [0.031893, -0.064069, 0.006475],
            [0.002485, 0.002021, 0.031893],
            [-0.002856, -0.001725, -0.035089],
            [-0.046640, -0.035089, 0.002457],
            [0.002318, -0.002856, -0.046640],
            [-0.063134, -0.013966, 0.001529],
            [0.002180, 0.001529, 0.027345],
            [0.088583, -0.063134, 0.002180],
            [-0.058656, -0.038783, -0.001542],
            [0.001772, -0.001542, -0.045197],
            [-0.084546, -0.058656, 0.001772],
        ]
    )

    compare_mmd(cgtoi, cgtoj, vec, ovlp_ref, ovlp_grad_ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_overlap_grad_single_6(dtype: torch.dtype):
    """Comparison of single gaussians. Reference values taken from tblite MMD implementation."""

    # define CGTOs (by ng, n, l)
    cgtoi = torch.tensor([6, 3, 2])
    cgtoj = torch.tensor([6, 5, 4])
    vec = -torch.tensor([1.245, -6.789, 0.123], dtype=dtype)

    # define references
    ovlp_ref = torch.tensor(
        [
            [
                -0.174043,
                0.003607,
                -0.019667,
                -0.157564,
                -0.059801,
                0.002831,
                -0.004679,
                0.103841,
                0.092087,
            ],
            [
                0.002050,
                -0.103968,
                -0.041379,
                0.007470,
                -0.013065,
                -0.238646,
                -0.167092,
                0.006860,
                -0.007296,
            ],
            [
                -0.011178,
                -0.041379,
                0.114085,
                -0.011003,
                -0.007092,
                -0.100442,
                0.063036,
                0.000028,
                -0.002428,
            ],
            [
                0.047791,
                0.001188,
                0.001725,
                0.027814,
                -0.011216,
                0.006080,
                -0.002480,
                -0.088792,
                -0.151113,
            ],
            [
                0.018138,
                -0.003936,
                -0.000150,
                -0.011216,
                0.053109,
                -0.007797,
                -0.007055,
                -0.141994,
                0.064765,
            ],
        ]
    )
    ovlp_grad_ref = torch.tensor(
        [
            [0.008758, -0.047758, 0.012878],
            [0.002624, 0.001490, 0.029216],
            [0.001490, -0.005228, -0.159313],
            [0.024609, -0.038129, 0.010932],
            [-0.042036, -0.023893, 0.004149],
            [0.002056, 0.000061, 0.022921],
            [0.001117, 0.000728, -0.037884],
            [-0.059295, 0.027474, 0.000470],
            [0.053158, 0.043755, 0.000416],
            [0.001613, 0.000183, 0.016599],
            [0.018041, -0.031904, 0.005240],
            [-0.031904, -0.001170, 0.003332],
            [0.004948, 0.002842, 0.060575],
            [0.004141, -0.004883, -0.105984],
            [0.100237, -0.063289, 0.007415],
            [-0.104239, -0.057263, 0.005564],
            [0.003918, 0.002052, 0.055685],
            [0.004618, -0.001193, -0.059230],
            [0.000183, 0.000650, -0.090516],
            [-0.031904, -0.001171, 0.003332],
            [-0.001170, -0.060090, -0.012319],
            [0.002842, 0.001894, -0.089154],
            [-0.004883, -0.001544, -0.057489],
            [-0.063289, 0.001540, 0.004050],
            [-0.057263, -0.063981, -0.003426],
            [0.002052, 0.002121, 0.000244],
            [-0.001193, 0.000737, -0.019701],
            [-0.009004, 0.019962, 0.000481],
            [0.000966, -0.000330, 0.009606],
            [-0.000330, 0.002995, 0.014093],
            [0.004199, 0.013140, 0.001527],
            [-0.009974, 0.013752, 0.002842],
            [0.003237, 0.000851, 0.049348],
            [0.004638, 0.000696, -0.020129],
            [0.134772, -0.021305, 0.000899],
            [-0.068194, -0.017450, 0.001940],
            [0.012166, 0.010434, 0.000183],
            [0.000422, -0.000270, -0.031904],
            [-0.000270, 0.001082, -0.001170],
            [-0.009974, 0.013752, 0.002842],
            [-0.002354, -0.023201, -0.004883],
            [0.004796, 0.000616, -0.063289],
            [-0.003991, -0.001009, -0.057263],
            [-0.068647, 0.023615, 0.002052],
            [-0.112938, -0.083108, -0.001193],
        ]
    )
    compare_mmd(cgtoi, cgtoj, vec, ovlp_ref, ovlp_grad_ref, dtype)
