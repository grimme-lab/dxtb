"""Run tests for overlap."""
from __future__ import annotations

import pytest
import torch
from torch.autograd.gradcheck import gradcheck, gradgradcheck
from torch.autograd.functional import jacobian


from dxtb.basis import slater
from dxtb.integral import mmd
from dxtb.typing import Tensor
from dxtb.basis import IndexHelper
from dxtb.integral import Overlap
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular

from .samples import samples
from .gradients import gradients


# tmp
from dxtb.integral.overlap import get_ovlp_grad


def test_gradcheck(dtype: torch.dtype = torch.double) -> None:

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
        mmd.EFunction.apply,
        (xij.clone().requires_grad_(True), rpi, rpj, shape),  # type: ignore
    )
    assert gradgradcheck(
        mmd.EFunction.apply,
        (xij.clone().requires_grad_(True), rpi, rpj, shape),  # type: ignore
    )

    assert gradcheck(
        mmd.EFunction.apply,
        (
            xij,
            rpi.clone().requires_grad_(True),
            rpj.clone().requires_grad_(True),
            shape,
        ),  # type: ignore
    )
    assert gradgradcheck(
        mmd.EFunction.apply,
        (
            xij,
            rpi.clone().requires_grad_(True),
            rpj.clone().requires_grad_(True),
            shape,
        ),  # type: ignore
    )


def test_gradcheck_overlap(dtype: torch.dtype = torch.double) -> None:
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
        mmd.overlap,
        (angular, alpha, coeff, vec.clone().requires_grad_(True)),  # type: ignore
    )
    assert gradgradcheck(
        mmd.overlap,
        (angular, alpha, coeff, vec.clone().requires_grad_(True)),  # type: ignore
    )


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
    vec = rndm.detach().clone().requires_grad_(True)
    s = mmd.overlap((l1, l2), (alpha1, alpha2), (coeff1, coeff2), vec)

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
        sr = mmd.overlap((l1, l2), (alpha1, alpha2), (coeff1, coeff2), rndm)

        rndm[i] -= 2 * step
        sl = mmd.overlap((l1, l2), (alpha1, alpha2), (coeff1, coeff2), rndm)

        rndm[i] += step
        g = 0.5 * (sr - sl) / step

        assert pytest.approx(gradient[i], abs=tol) == g.item()
        assert pytest.approx(gradient[i], abs=tol) == ref[i]


# TODO:
# 1. compare with numerical gradient
# 2. compare with tblite gradient
#   a. get tblite cgto_grad running
#   b. apply tblite on samples
#   c. create reference  data with tblite overlap_grad_cgto (has also shape [norb, norb, 3])
# 3. compare with AD gradient
# 4. get shape right
# 5. check that samples with l>1 work
# 6. implement batch mode
# (7. make vectorisible)
# 8. write tests
#   a. SiH4 is quite dodgy
# 9. check RHF / UHF functional
# 10. reimplement dS in SHARK fashion: doing atomwise loop

# TODO:
# 1. test for different number of gaussian ng


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
# @pytest.mark.parametrize("name", ["H", "C", "Rn"])
@pytest.mark.parametrize("name", ["H2O", "CH4", "SiH4", "PbH4-BiH3"])
def test_overlap_grad_single(dtype: torch.dtype, name: str):
    """Gradient of overlap matrix for monoatomic molecule."""
    dd = {"dtype": dtype}
    tol = 1e-05

    print("\nName", name)

    # reference gradient (tblite non-MMD)
    ref = gradients[name]

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    overlap = Overlap(numbers, par, ihelp, **dd)

    # analytical gradient
    dsdr = overlap.get_gradient(positions)  # [norb, norb, 3] # [natm, 3]

    # autograd gradient
    positions.requires_grad_(True)
    s = overlap.build(positions)  # [norb, norb]
    gradient = torch.autograd.grad(
        s,
        positions,
        grad_outputs=torch.ones_like(s),
    )[0]

    print("analytical gradient \n", dsdr)

    print("autograd gradient \n", gradient)

    print("tblite gradient\n", ref)
    print(ref.shape)

    assert dsdr.shape == gradient.shape
    assert ref.shape == gradient.shape

    # tmp (wrong) result
    tmp_ref = torch.tensor(
        [
            [-2.0286331177, -3.1291103363, -1.4172217846],
            [2.2608838081, 0.5714894533, 0.1429068148],
            [-2.2183816433, 0.5714894533, -0.7118155956],
        ]
    ).double()
    assert torch.allclose(dsdr, tmp_ref)

    print("Finish")
    return
    assert pytest.approx(s, rel=tol, abs=tol) == ref

    ### Check dependency on stepsize ###
    # NOTE: dytpe.float() | double()
    # H2O
    # -1 | 0.2239793092 | 0.2239793254
    # -2 | 0.2245728970 | 0.2245728800
    # -4 | 0.2245788723 | 0.2245788856
    # -6 | 0.2733475566 | 0.2245788862
    # SiH4
    # -1 | 0.1862625033 | 0.1862624826
    # -2 | 0.1865297556 | 0.1865306996
    # -4 | 0.1867115498 | 0.1865334097
    # -6 | 0.1862645149 | 0.1865334100
    # CH4
    # -1 | 0.0002828687 | 0.0002828764
    # -2 | 0.0000282712 | 0.0000028310
    # -4 | 0.0025108866 | 0.0000000003 <-- use as reference for development
    # -6 | 0.5723313093 | 0.0000000006

    ## pytest -ssv test/test_overlap/test_grad.py::test_overlap_grad_single[CH4-dtype1] --tb=no


def compare_mmd(
    cgtoi: Tensor,
    cgtoj: Tensor,
    vec: Tensor,
    ovlp_ref: Tensor,
    ovlp_grad_ref: Tensor,
    dtype: torch.dtype,
) -> None:
    """Helper method to compare MMD overlap and gradient with references.

    Args:
        cgtoi (Tensor): Specification for first CGTO, containing ng, n, l
        cgtoj (Tensor): Specification for second CGTO, containing ng, n, l
        vec (Tensor): Shift vector of two CGTO centers
        ovlp_ref (Tensor): Reference for overlap value
        ovlp_grad_ref (Tensor): Reference for overlap gradient value
        dtype (torch.dtype): Dtype for tensors
    """

    # define tolerances
    atol = 1.0e-5

    ngi, ni, li = cgtoi
    ngj, nj, lj = cgtoj
    alpha_i, coeff_i = slater.to_gauss(ngi, ni, li, torch.tensor(1.0, dtype=dtype))
    alpha_j, coeff_j = slater.to_gauss(ngj, nj, lj, torch.tensor(1.0, dtype=dtype))

    # overlap
    ovlp = mmd.overlap((li, lj), (alpha_i, alpha_j), (coeff_i, coeff_j), vec)
    print("ovlp\n", ovlp, ovlp.shape)
    print("ovlp_ref\n", ovlp_ref, ovlp_ref.shape)
    assert pytest.approx(ovlp, abs=atol) == ovlp_ref

    # overlap gradient
    ovlp_grad = get_ovlp_grad((li, lj), (alpha_i, alpha_j), (coeff_i, coeff_j), vec)

    # obtain Fortran ordering (row wise)
    ovlp_grad = torch.stack([ovlp_grad[i].flatten() for i in range(3)]).transpose(0, 1)

    print("ovlp_grad\n", ovlp_grad, ovlp_grad.shape)
    print("ovlp_grad_ref\n", ovlp_grad_ref, ovlp_grad_ref.shape)
    assert pytest.approx(ovlp_grad, abs=atol) == ovlp_grad_ref


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_overlap_grad_single_1(dtype: torch.dtype):
    """Comparison of single gaussians. Reference values taken from tblite MMD implementation."""

    # define CGTOs (by ng, n, l)
    cgtoi = torch.tensor([6, 2, 0])
    cgtoj = torch.tensor([6, 1, 0])
    vec = torch.tensor([0.00, 0.00, 1.405], dtype=dtype)

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
    vec = torch.tensor([0.00, 0.00, 1.405], dtype=dtype)

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
    vec = torch.tensor([6.2571, -3.5633, -2.5369], dtype=dtype)

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
    vec = torch.tensor([1.245, -6.789, 0.123], dtype=dtype)

    # define references
    ovlp_ref = torch.tensor(
        [
            [0.086376, -0.005569, -0.140234, -0.004323, -0.068378, -0.005745, 0.094314],
            [-0.013510, -0.028473, -0.006903, -0.149172, 0.001266, -0.028473, 0.002478],
            [-0.055729, -0.007895, -0.180163, 0.003929, -0.102090, 0.003369, -0.186877],
        ]
    ).reshape([7, 3])
    ovlp_grad_ref = torch.tensor(
        [
            [
                -0.058656,
                -0.038783,
                -0.001542,
                0.001772,
                -0.001542,
                -0.045197,
                -0.084546,
            ],
            [
                -0.058656,
                0.001772,
                -0.002856,
                -0.001725,
                -0.035089,
                -0.046640,
                -0.035089,
            ],
            [0.002457, 0.002318, -0.002856, -0.046640, -0.004884, -0.019109, -0.005973],
            [0.001460, -0.005973, -0.109565, -0.021205, -0.004884, 0.001460, 0.000405],
            [-0.001190, -0.055980, 0.010266, -0.055980, 0.006906, 0.000943, 0.000405],
            [0.010266, -0.021205, -0.004884, 0.001460, 0.001722, 0.001460, 0.020093],
            [0.012277, -0.021205, 0.001722, 0.002021, -0.000921, -0.064069, 0.031893],
            [-0.064069, 0.006475, 0.002485, 0.002021, 0.031893, -0.063134, -0.013966],
            [0.001529, 0.002180, 0.001529, 0.027345, 0.088583, -0.063134, 0.002180],
        ]
    ).reshape([21, 3])

    # TODO: permute dxtb order such that it matches the tblite order. then do similar printout in fortran/python and compare

    # ovlp [3, 7, 3]
    # [ 0.0004, -0.0012, -0.0560], <- 4th
    # [ 0.0103, -0.0560,  0.0069],
    # [ 0.0009,  0.0004,  0.0103],
    # [-0.0212, -0.0049,  0.0015], <- 3rd to last
    # [ 0.0017,  0.0015,  0.0201],
    # [ 0.0123, -0.0212,  0.0017],
    # [-0.0049, -0.0191, -0.0060], <- 3rd
    # [ 0.0015, -0.0060, -0.1096],
    # [-0.0212, -0.0049,  0.0015],
    # [ 0.0020, -0.0009, -0.0641], <- 2nd to last
    # [ 0.0319, -0.0641,  0.0065],
    # [ 0.0025,  0.0020,  0.0319],
    # [-0.0029, -0.0017, -0.0351], <- 2nd
    # [-0.0466, -0.0351,  0.0025],
    # [ 0.0023, -0.0029, -0.0466],
    # [-0.0631, -0.0140,  0.0015], <- last
    # [ 0.0022,  0.0015,  0.0273],
    # [ 0.0886, -0.0631,  0.0022],
    # [-0.0587, -0.0388, -0.0015], <- 1st
    # [ 0.0018, -0.0015, -0.0452],
    # [-0.0845, -0.0587,  0.0018]]) torch.Size([21, 3])
    # ovlp_grad_ref
    # [-0.0587, -0.0388, -0.0015],
    # [ 0.0018, -0.0015, -0.0452],
    # [-0.0845, -0.0587,  0.0018],
    # [-0.0029, -0.0017, -0.0351],
    # [-0.0466, -0.0351,  0.0025],
    # [ 0.0023, -0.0029, -0.0466],
    # [-0.0049, -0.0191, -0.0060],
    # [ 0.0015, -0.0060, -0.1096],
    # [-0.0212, -0.0049,  0.0015],
    # [ 0.0004, -0.0012, -0.0560],
    # [ 0.0103, -0.0560,  0.0069],
    # [ 0.0009,  0.0004,  0.0103],
    # [-0.0212, -0.0049,  0.0015],
    # [ 0.0017,  0.0015,  0.0201],
    # [ 0.0123, -0.0212,  0.0017],
    # [ 0.0020, -0.0009, -0.0641],
    # [ 0.0319, -0.0641,  0.0065],
    # [ 0.0025,  0.0020,  0.0319],
    # [-0.0631, -0.0140,  0.0015],
    # [ 0.0022,  0.0015,  0.0273],
    # [ 0.0886, -0.0631,  0.0022]]) torch.Size([21, 3])

    compare_mmd(cgtoi, cgtoj, vec, ovlp_ref, ovlp_grad_ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_overlap_grad_single_6(dtype: torch.dtype):
    """Comparison of single gaussians. Reference values taken from tblite MMD implementation."""

    # define CGTOs (by ng, n, l)
    cgtoi = torch.tensor([6, 3, 2])
    cgtoj = torch.tensor([6, 5, 4])
    vec = torch.tensor([1.245, -6.789, 0.123], dtype=dtype)

    torch.set_printoptions(precision=6, linewidth=300, sci_mode=False)

    # TODO: generally the sorting is not identical (neither of ovlp nor of ovlp_grad)
    #      apply same ordering scheme as in ovlp from overlap to gradient

    # ovlp (normal dxtb mmd)
    #     [    -0.174043,      0.003607,     -0.019667,     -0.157564,     -0.059801,      0.002831,     -0.004679,      0.103841,      0.092087],
    #     [     0.002050,     -0.103968,     -0.041379,      0.007470,     -0.013065,     -0.238646,     -0.167092,      0.006860,     -0.007296],
    #     [    -0.011178,     -0.041379,      0.114085,     -0.011003,     -0.007092,     -0.100442,      0.063036,      0.000028,     -0.002428],
    #     [     0.047791,      0.001188,      0.001725,      0.027814,     -0.011216,      0.006080,     -0.002480,     -0.088792,     -0.151113],
    #     [     0.018138,     -0.003936,     -0.000150,     -0.011216,      0.053109,     -0.007797,     -0.007055,     -0.141994,      0.064765]
    #     torch.Size([5, 9])
    # ovlp_ref
    #     [     0.064765,     -0.007055,      0.053109,     -0.000150,      0.018138,     -0.003936,     -0.011216,     -0.007797,     -0.141994],
    #     [    -0.002428,      0.063036,     -0.007092,      0.114085,     -0.011178,     -0.041379,     -0.011003,     -0.100442,      0.000028],
    #     [     0.092087,     -0.004679,     -0.059801,     -0.019667,     -0.174043,      0.003607,     -0.157565,      0.002831,      0.103841],
    #     [    -0.007296,     -0.167092,     -0.013065,     -0.041379,      0.002050,     -0.103968,      0.007470,     -0.238646,      0.006860],
    #     [    -0.151113,     -0.002480,     -0.011216,      0.001725,      0.047791,      0.001188,      0.027814,      0.006080,     -0.088792]]
    #     torch.Size([5, 9])

    # define references
    ovlp_ref = torch.tensor(
        [
            [0.064765, -0.007055, 0.053109, -0.000150, 0.018138],
            [-0.003936, -0.011216, -0.007797, -0.141994, -0.002428],
            [0.063036, -0.007092, 0.114085, -0.011178, -0.041379],
            [-0.011003, -0.100442, 0.000028, 0.092087, -0.004679],
            [-0.059801, -0.019667, -0.174043, 0.003607, -0.157565],
            [0.002831, 0.103841, -0.007296, -0.167092, -0.013065],
            [-0.041379, 0.002050, -0.103968, 0.007470, -0.238646],
            [0.006860, -0.151113, -0.002480, -0.011216, 0.001725],
            [0.047791, 0.001188, 0.027814, 0.006080, -0.088792],
        ]
    ).reshape([5, 9])
    ovlp_grad_ref = torch.tensor(
        [
            [-0.112938, -0.083108, -0.001193, -0.003991, -0.001009],
            [-0.057263, -0.002354, -0.023201, -0.004883, -0.000270],
            [0.001082, -0.001171, 0.012166, 0.010434, 0.000183],
            [0.000422, -0.000270, -0.031904, -0.009974, 0.013752],
            [0.002842, 0.004796, 0.000616, -0.063289, -0.068647],
            [0.023615, 0.002052, -0.001193, 0.000737, -0.019701],
            [-0.057263, -0.063981, -0.003426, -0.004883, -0.001544],
            [-0.057489, -0.001171, -0.060090, -0.012319, 0.000183],
            [0.000650, -0.090516, -0.031904, -0.001171, 0.003332],
            [0.002842, 0.001894, -0.089155, -0.063289, 0.001540],
            [0.004050, 0.002052, 0.002121, 0.000244, 0.053158],
            [0.043755, 0.000416, 0.001117, 0.000728, -0.037884],
            [-0.042036, -0.023893, 0.004149, 0.001490, -0.005228],
            [-0.159313, 0.008758, -0.047758, 0.012878, 0.002624],
            [0.001490, 0.029216, 0.024609, -0.038129, 0.010932],
            [0.002056, 0.000061, 0.022921, -0.059295, 0.027474],
            [0.000470, 0.004618, -0.001193, -0.059230, -0.104239],
            [-0.057263, 0.005564, 0.004141, -0.004883, -0.105984],
            [-0.031904, -0.001171, 0.003332, 0.001613, 0.000183],
            [0.016599, 0.018041, -0.031904, 0.005240, 0.004948],
            [0.002842, 0.060575, 0.100237, -0.063289, 0.007415],
            [0.003918, 0.002052, 0.055685, -0.068194, -0.017450],
            [0.001940, 0.004638, 0.000696, -0.020129, -0.009974],
            [0.013752, 0.002842, -0.000330, 0.002995, 0.014093],
            [-0.009004, 0.019962, 0.000481, 0.000966, -0.000330],
            [0.009606, 0.004199, 0.013140, 0.001527, 0.003237],
            [0.000851, 0.049349, 0.134772, -0.021305, 0.000899],
        ]
    )  # .reshape([25, 3])

    compare_mmd(cgtoi, cgtoj, vec, ovlp_ref, ovlp_grad_ref, dtype)
