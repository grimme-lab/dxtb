"""Run tests for overlap."""

from math import sqrt

import pytest
import torch
from torch.autograd.gradcheck import gradcheck, gradgradcheck

from dxtb.basis import slater
from dxtb.integral import mmd


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
        **dd
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
