"""
Test CGTO normalization.
"""

from __future__ import annotations

import pytest
import torch

from dxtb.basis import slater_to_gauss
from dxtb.integral.driver.pytorch.impls.md import overlap_gto


@pytest.mark.parametrize("ng", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "n, l",
    [
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (3, 2),
        (4, 2),
        (5, 2),
        (4, 3),
        (5, 3),
        # (5, 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_sto_ng_single(ng, n, l, dtype):
    """
    Test normalization of all STO-NG basis functions
    """
    atol = 1.0e-6 if dtype == torch.float else 2.0e-7

    alpha, coeff = slater_to_gauss(ng, n, l, torch.tensor(1.0, dtype=dtype))
    angular = torch.tensor(l)
    vec = torch.zeros((3,), dtype=dtype)

    s = overlap_gto((angular, angular), (alpha, alpha), (coeff, coeff), vec)
    ref = torch.diag(torch.ones((2 * l + 1,), dtype=dtype))

    assert pytest.approx(ref, abs=atol) == s


@pytest.mark.parametrize("ng", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_sto_ng_batch(ng: int, dtype: torch.dtype):
    """
    Test symmetry of s integrals
    """
    n, l = torch.tensor(1), torch.tensor(0)
    ng_ = torch.tensor(ng)

    coeff, alpha = slater_to_gauss(ng_, n, l, torch.tensor(1.0, dtype=dtype))
    coeff, alpha = coeff.type(dtype)[:ng_], alpha.type(dtype)[:ng_]
    vec = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)

    s = overlap_gto((l, l), (alpha, alpha), (coeff, coeff), vec)

    assert pytest.approx(s[0, :]) == s[1, :]
    assert pytest.approx(s[0, :]) == s[2, :]


@pytest.mark.parametrize("ng", [6])
@pytest.mark.parametrize("n, l", [(1, 0)])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_no_norm(ng, n, l, dtype):
    """
    Test normalization of all STO-NG basis functions
    """
    tol = 1.0e-7

    ref_alpha = torch.tensor(
        [
            23.1030311584,
            4.2359156609,
            1.1850565672,
            0.4070988894,
            0.1580884159,
            0.0651095361,
        ]
    )

    ref_coeff = torch.tensor(
        [
            0.0091635967,
            0.0493614934,
            0.1685383022,
            0.3705627918,
            0.4164915383,
            0.1303340793,
        ]
    )

    zeta = torch.tensor(1.0, dtype=dtype)
    alpha, coeff = slater_to_gauss(ng, n, l, zeta, norm=False)

    assert pytest.approx(ref_alpha, abs=tol, rel=tol) == alpha
    assert pytest.approx(ref_coeff, abs=tol, rel=tol) == coeff
