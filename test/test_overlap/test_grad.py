"""
Run tests for overlap gradient.
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch
from torch.autograd.functional import jacobian

from dxtb._types import DD, Tensor
from dxtb.basis import IndexHelper, slater_to_gauss
from dxtb.integral.driver.pytorch import IntDriverPytorch as IntDriver
from dxtb.integral.driver.pytorch import OverlapPytorch as Overlap
from dxtb.integral.driver.pytorch.impls import md
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import t2int

from .samples import samples

device = None


@pytest.mark.parametrize("dtype", [torch.double])
def test_ss(dtype: torch.dtype):
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    ng = torch.tensor(6)
    n1, n2 = torch.tensor(2), torch.tensor(1)
    l1, l2 = torch.tensor(0), torch.tensor(0)

    alpha1, coeff1 = slater_to_gauss(ng, n1, l1, torch.tensor(1.0, dtype=dtype))
    alpha2, coeff2 = slater_to_gauss(ng, n2, l2, torch.tensor(1.0, dtype=dtype))

    rndm = torch.tensor(
        [0.13695892585203528, 0.47746994997214642, 0.20729096231197164], **dd
    )
    vec = rndm.detach().requires_grad_(True)
    s = md.overlap_gto((l1, l2), (alpha1, alpha2), (coeff1, coeff2), vec)

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
        sr = md.overlap_gto((l1, l2), (alpha1, alpha2), (coeff1, coeff2), rndm)

        rndm[i] -= 2 * step
        sl = md.overlap_gto((l1, l2), (alpha1, alpha2), (coeff1, coeff2), rndm)

        rndm[i] += step
        g = 0.5 * (sr - sl) / step

        assert pytest.approx(gradient[i], abs=tol) == g.item()
        assert pytest.approx(gradient[i], abs=tol) == ref[i]


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["H", "H2", "LiH", "C", "Rn", "H2O", "CH4", "SiH4"])
def test_overlap_jacobian(dtype: torch.dtype, name: str):
    """Jacobian calculation with AD and numerical gradient."""
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps)

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    driver = IntDriver(numbers, par, ihelp, **dd)
    overlap = Overlap(uplo="n", **dd)

    # numerical gradient
    ngrad = calc_numerical_gradient(overlap, driver, positions)  # [natm, norb, norb, 3]

    # autograd jacobian
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        driver.setup(pos)
        return overlap.build(driver)

    # [norb, norb, natm, 3]
    # j = jacobian(func, argnums=0)(positions)
    j = jacobian(func, positions)
    j = torch.movedim(j, -2, 0)  # type: ignore

    # check whether dimensions are swapped
    assert torch.equal(ngrad - j, ngrad - torch.movedim(j, 1, 2))

    assert pytest.approx(ngrad, rel=tol, abs=tol) == j

    positions.detach_()


def calc_numerical_gradient(
    overlap: Overlap, driver: IntDriver, positions: Tensor
) -> Tensor:
    step = 1.0e-4
    natm = positions.shape[0]
    norb = t2int(driver.ihelp.orbitals_per_shell.sum())
    gradient = positions.new_zeros((natm, norb, norb, 3))

    # coordinates shift preferred to orbitals shift
    for i in range(natm):
        for j in range(3):
            positions[i, j] += step
            driver.setup(positions)
            sr = overlap.build(driver)  # [norb, norb]

            positions[i, j] -= 2 * step
            driver.setup(positions)
            sl = overlap.build(driver)

            positions[i, j] += step
            gradient[i, :, :, j] = 0.5 * (sr - sl) / step

    return gradient


def compare_md(
    cgtoi: Tensor,
    cgtoj: Tensor,
    vec: Tensor,
    ovlp_ref: Tensor,
    ovlp_grad_ref: Tensor,
    dtype: torch.dtype,
) -> None:
    """Helper method to compare MD overlap and gradient with references.
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
    alpha_i, coeff_i = slater_to_gauss(ngi, ni, li, torch.tensor(1.0, dtype=dtype))
    alpha_j, coeff_j = slater_to_gauss(ngj, nj, lj, torch.tensor(1.0, dtype=dtype))

    # overlap
    ovlp = md.overlap_gto((li, lj), (alpha_i, alpha_j), (coeff_i, coeff_j), vec)
    assert pytest.approx(ovlp, abs=atol) == ovlp_ref

    # overlap gradient with explicit E-coefficients
    ovlp_grad_exp = md.explicit.md_explicit_gradient(
        (li, lj), (alpha_i, alpha_j), (coeff_i, coeff_j), vec
    )

    # overlap gradient with recursion
    ovlp_grad_rec = md.recursion.md_recursion_gradient(
        (li, lj), (alpha_i, alpha_j), (coeff_i, coeff_j), vec
    )
    ovlp_grad_rec = torch.squeeze(ovlp_grad_rec, 0)

    # obtain Fortran ordering (row wise)
    ovlp_grad_rec = torch.stack(
        [ovlp_grad_rec[i].flatten() for i in range(3)]
    ).transpose(0, 1)
    ovlp_grad_exp = torch.stack(
        [ovlp_grad_exp[i].flatten() for i in range(3)]
    ).transpose(0, 1)

    assert pytest.approx(ovlp_grad_exp, abs=atol) == ovlp_grad_rec
    assert pytest.approx(ovlp_grad_ref, abs=atol) == ovlp_grad_rec
    assert pytest.approx(ovlp_grad_ref, abs=atol) == ovlp_grad_exp


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_overlap_grad_single_1s1s(dtype: torch.dtype):
    """
    Comparison of single gaussians. Reference values taken from tblite MD
    implementation.
    """

    # define CGTOs (by ng, n, l)
    cgtoi = torch.tensor([6, 1, 0])  # 1s
    cgtoj = torch.tensor([6, 1, 0])  # 1s
    vec = -torch.tensor([0.00, 0.00, 1.405], dtype=dtype)

    # define references
    ovlp_ref = torch.tensor([[0.75156098243428548]])
    ovlp_grad_ref = torch.tensor(
        [[0.0000000000000000, 0.0000000000000000, -0.27637559271358614]]
    )

    compare_md(cgtoi, cgtoj, vec, ovlp_ref, ovlp_grad_ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_overlap_grad_single_1s2s(dtype: torch.dtype):
    """
    Comparison of single gaussians. Reference values taken from tblite MD
    implementation.
    """

    # define CGTOs (by ng, n, l)
    cgtoi = torch.tensor([6, 2, 0])  # 2s
    cgtoj = torch.tensor([6, 1, 0])  # 1s
    vec = -torch.tensor([0.00, 0.00, 1.405], dtype=dtype)

    # define references
    ovlp_ref = torch.tensor([[0.76296162997970662]])
    ovlp_grad_ref = torch.tensor(
        [0.0000000000000000, 0.0000000000000000, -0.14526133860726220]
    ).reshape([1, 3])

    compare_md(cgtoi, cgtoj, vec, ovlp_ref, ovlp_grad_ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_overlap_grad_single_3s4s(dtype: torch.dtype):
    """
    Comparison of single gaussians.
    Reference values taken from tblite MD implementation.
    """

    # define CGTOs (by ng, n, l)
    cgtoi = torch.tensor([6, 3, 0])
    cgtoj = torch.tensor([6, 4, 0])
    vec = -torch.tensor([0.00, 0.00, 1.405], dtype=dtype)

    # define references
    ovlp_ref = torch.tensor([[0.893829]])
    ovlp_grad_ref = torch.tensor([0.000000, 0.000000, -0.056111]).reshape([1, 3])

    compare_md(cgtoi, cgtoj, vec, ovlp_ref, ovlp_grad_ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_overlap_grad_single_1s2p(dtype: torch.dtype):
    """
    Comparison of single gaussians.
    Reference values taken from tblite MD
    implementation.
    """

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

    compare_md(cgtoi, cgtoj, vec, ovlp_ref, ovlp_grad_ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float])
def test_overlap_grad_single_2p2p(dtype: torch.dtype):
    """
    Comparison of single gaussians. Reference values taken from tblite MD
    implementation.
    """

    # define CGTOs (by ng, n, l)
    cgtoi = torch.tensor([6, 2, 1])  # 2p
    cgtoj = torch.tensor([6, 2, 1])  # 2p
    vec = torch.tensor([0.00, 0.00, 1.405], dtype=dtype)

    # define references
    ovlp_ref = torch.tensor(
        [
            [
                +8.2922059116650637e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +5.3250035739126811e-01,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +8.2922059116650637e-01,
            ],
        ]
    )
    ovlp_grad_ref = torch.tensor(
        [
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +2.1118877848771400e-01,
            ],
            [
                +0.0000000000000000e00,
                +2.1118877848771400e-01,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +2.1118877848771400e-01,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +5.2445032996407726e-01,
            ],
            [
                +2.1118877848771400e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +2.1118877848771400e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +2.1118877848771400e-01,
            ],
        ]
    )

    compare_md(cgtoi, cgtoj, vec, ovlp_ref, ovlp_grad_ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float])
def test_overlap_grad_single_3d3d(dtype: torch.dtype):
    """
    Comparison of single gaussians. Reference values taken from tblite MD
    implementation.
    """

    # define CGTOs (by ng, n, l)
    cgtoi = torch.tensor([6, 3, 2])  # 3d
    cgtoj = torch.tensor([6, 3, 2])  # 3d
    vec = torch.tensor([0.00, 0.00, 1.405], dtype=dtype)

    # define references
    ovlp_ref = torch.tensor(
        [
            [
                +5.8771008310057316e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +6.3776604042828389e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +6.3776604042828389e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +8.7160919992189456e-01,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +8.7160919992189512e-01,
            ],
        ]
    )
    ovlp_grad_ref = torch.tensor(
        [
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +4.6170905976008525e-01,
            ],
            [
                +6.1707801646330884e-02,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +6.1707801646330884e-02,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +6.1707801646330912e-02,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +4.3975387257676468e-01,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +1.6643641245096857e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +1.6643641245096863e-01,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +6.1707801646330912e-02,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +4.3975387257676468e-01,
            ],
            [
                +0.0000000000000000e00,
                -1.6643641245096857e-01,
                +0.0000000000000000e00,
            ],
            [
                +1.6643641245096863e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +1.6643641245096866e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                -1.6643641245096866e-01,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +1.6643641245096863e-01,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +1.6643641245096857e-01,
                +0.0000000000000000e00,
            ],
            [
                +1.6643641245096857e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +1.6643641245096855e-01,
            ],
        ]
    )

    compare_md(cgtoi, cgtoj, vec, ovlp_ref, ovlp_grad_ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float])
def test_overlap_grad_single_3d4d(dtype: torch.dtype):
    """
    Comparison of single gaussians. Reference values taken from tblite MD
    implementation.
    """

    # define CGTOs (by ng, n, l)
    cgtoi = torch.tensor([6, 4, 2])  # 3d
    cgtoj = torch.tensor([6, 3, 2])  # 3d
    vec = torch.tensor([0.00, 0.00, 1.405], dtype=dtype)

    # define references
    ovlp_ref = torch.tensor(
        [
            [
                +6.3643879421504990e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +6.7913962944500306e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +6.7913962944500306e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +8.4656380658515418e-01,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +8.4656380658515373e-01,
            ],
        ]
    )
    ovlp_grad_ref = torch.tensor(
        [
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +3.6479506604265466e-01,
            ],
            [
                +5.2640580885342528e-02,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +5.2640580885342528e-02,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +5.2640580885342493e-02,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +3.2950239143783705e-01,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +1.1916311540224239e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +1.1916311540224236e-01,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +5.2640580885342493e-02,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +3.2950239143783705e-01,
            ],
            [
                +0.0000000000000000e00,
                -1.1916311540224239e-01,
                +0.0000000000000000e00,
            ],
            [
                +1.1916311540224236e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +1.1916311540224241e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                -1.1916311540224241e-01,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +1.1916311540224239e-01,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +1.1916311540224234e-01,
                +0.0000000000000000e00,
            ],
            [
                +1.1916311540224234e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +1.1916311540224236e-01,
            ],
        ]
    )

    compare_md(cgtoi, cgtoj, vec, ovlp_ref, ovlp_grad_ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float])
def test_overlap_grad_single_4f4f(dtype: torch.dtype):
    """
    Comparison of single gaussians. Reference values taken from tblite MD
    implementation.
    """

    # define CGTOs (by ng, n, l)
    cgtoi = torch.tensor([6, 4, 3])  # 4f
    cgtoj = torch.tensor([6, 4, 3])  # 4f
    vec = torch.tensor([0.00, 0.00, 1.405], dtype=dtype)

    # define references
    ovlp_ref = torch.tensor(
        [
            [
                +8.9759172356121597e-01,
                +0.0000000000000000e00,
                -1.9428902930940239e-16,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +7.0641635946489045e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                -1.2490009027033011e-16,
                +0.0000000000000000e00,
                +6.1222729334880099e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +5.8302669766420001e-01,
                +0.0000000000000000e00,
                +2.7755575615628914e-17,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +6.1222729334880088e-01,
                +0.0000000000000000e00,
                +1.9428902930940239e-16,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +7.0641635946489012e-01,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +1.9428902930940239e-16,
                +0.0000000000000000e00,
                +8.9759172356121597e-01,
            ],
        ]
    )
    ovlp_grad_ref = torch.tensor(
        [
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +1.3606787480165553e-01,
            ],
            [
                +1.6664843182448039e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +5.6094129952752161e-17,
            ],
            [
                +0.0000000000000000e00,
                +1.0408340855860843e-17,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +1.6664843182448039e-01,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +1.6664843182448041e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +3.7169801525602220e-01,
            ],
            [
                +1.0599714576904999e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +1.0599714576904999e-01,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                -1.6664843182448041e-01,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +3.4798296399259489e-17,
            ],
            [
                +1.0599714576905002e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +4.5989245885270613e-01,
            ],
            [
                +0.0000000000000000e00,
                +5.0908583354156198e-02,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                -1.0599714576904999e-01,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +1.2795192636143784e-17,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +5.0908583354156219e-02,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +4.8520233295033538e-01,
            ],
            [
                +5.0908583354156212e-02,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +4.8931385042321104e-18,
            ],
            [
                -1.6909450607132104e-17,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +1.0599714576905002e-01,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +5.0908583354156198e-02,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +4.5989245885270613e-01,
            ],
            [
                +1.0599714576904999e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                -3.4798296399259489e-17,
            ],
            [
                +0.0000000000000000e00,
                +1.6664843182448039e-01,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                -1.0599714576904998e-01,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                -1.3877787807814457e-17,
            ],
            [
                +1.0599714576904996e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +3.7169801525602236e-01,
            ],
            [
                +1.6664843182448039e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                -1.6664843182448039e-01,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                -1.3877787807814457e-17,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                -5.6094129952752161e-17,
            ],
            [
                +1.6664843182448039e-01,
                +0.0000000000000000e00,
                +0.0000000000000000e00,
            ],
            [
                +0.0000000000000000e00,
                +0.0000000000000000e00,
                +1.3606787480165553e-01,
            ],
        ]
    )

    compare_md(cgtoi, cgtoj, vec, ovlp_ref, ovlp_grad_ref, dtype)
