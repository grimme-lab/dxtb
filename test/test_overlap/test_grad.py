"""Run tests for overlap."""

from math import sqrt

import pytest
import torch

from dxtb.basis import slater
from dxtb.integral import mmd


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


@pytest.mark.parametrize("dtype", [torch.double])
def test_pp(dtype: torch.dtype):

    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    ng = torch.tensor(6)
    n1, n2 = torch.tensor(3), torch.tensor(2)
    l1, l2 = torch.tensor(1), torch.tensor(1)

    alpha1, coeff1 = slater.to_gauss(ng, n1, l1, torch.tensor(1.0, dtype=dtype))
    alpha2, coeff2 = slater.to_gauss(ng, n2, l2, torch.tensor(1.0, dtype=dtype))

    rndm = torch.tensor(
        [-0.32287465247083114, -0.33340160704754862, -0.43670797180952992], **dd
    )
    vec = rndm.detach().clone().requires_grad_(True)
    s = mmd.overlap((l1, l2), (alpha1, alpha2), (coeff1, coeff2), vec)

    print(s)

    # autograd
    gradient = torch.autograd.grad(
        s,
        vec,
        grad_outputs=torch.ones_like(s),
    )[0]

    print("\ngradient")
    print(gradient)
    print("\n")

    # reference gradient from tblite
    ref = torch.tensor(
        [
            [
                [
                    -6.8324049195113718e-03,
                    -1.2832018802516673e-01,
                    +5.2179824527632090e-02,
                ],
                [
                    -1.7217557611869829e-04,
                    +5.2179824527632090e-02,
                    -4.1745153775159360e-02,
                ],
                [
                    -4.3037532903789241e-02,
                    -6.8324049195113718e-03,
                    -1.7217557611869826e-04,
                ],
            ],
            [
                [
                    -1.7217557611869826e-04,
                    +5.2179824527632090e-02,
                    -4.1745153775159353e-02,
                ],
                [
                    -6.7587576032327899e-03,
                    -4.1745153775159360e-02,
                    +1.5810388993114854e-01,
                ],
                [
                    +5.3215382889574975e-02,
                    -1.7217557611869826e-04,
                    -6.7587576032327899e-03,
                ],
            ],
            [
                [
                    -4.3037532903789241e-02,
                    -6.8324049195113718e-03,
                    -1.7217557611869826e-04,
                ],
                [
                    +5.3215382889574982e-02,
                    -1.7217557611869829e-04,
                    -6.7587576032327908e-03,
                ],
                [
                    -2.0911301701840100e-02,
                    -4.3037532903789248e-02,
                    +5.3215382889574975e-02,
                ],
            ],
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

        print(g)
        print(gradient[i])
