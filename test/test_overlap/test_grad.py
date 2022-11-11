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
