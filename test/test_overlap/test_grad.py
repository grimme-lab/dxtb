"""Run tests for overlap."""
from __future__ import annotations

from math import sqrt

import pytest
import torch
from torch.autograd.gradcheck import gradcheck, gradgradcheck
from torch.autograd.functional import jacobian


from dxtb._types import Tensor
from dxtb.basis import slater
from dxtb.integral import mmd
from dxtb._types import Tensor
from dxtb.basis import IndexHelper
from dxtb.integral import Overlap
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular

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
    vec = rndm.detach().clone().requires_grad_(True)
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


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2O", "CH4", "SiH4", "PbH4-BiH3"])
def test_gradcheck_overlap(dtype: torch.dtype, name: str):
    """Pytorch gradcheck for overlap calculation."""
    dd = {"dtype": dtype}
    tol = 3e-01

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
        overlap = Overlap(numbers, par, ihelp, **dd)
        return overlap.build(pos)

    assert gradcheck(func, (positions), atol=tol)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H", "C", "Rn", "H2O", "CH4", "SiH4", "PbH4-BiH3"])
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
    positions.requires_grad_(False)  # turn off for multi parametrized tests

    # check whether dimensions are swapped
    assert torch.equal(ngrad - jac, ngrad - torch.movedim(jac, 1, 2))

    # jacobian and numerical gradient mismatch
    # print("diff", ngrad - jac)
    print("max diff", torch.max(ngrad - jac))

    assert pytest.approx(ngrad, rel=rtol, abs=atol) == jac


def calc_numerical_gradient(overlap: Overlap, positions: Tensor) -> Tensor:

    # setup numerical gradient
    step = 1.0e-4  # require larger deviations for overlap change
    natm = positions.shape[0]
    norb = overlap.ihelp.orbitals_per_shell.sum()
    gradient = torch.zeros((natm, norb, norb, 3), dtype=positions.dtype)

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
