import pytest
import torch

from xtbml.integral import mmd
from xtbml.basis import slater


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
        (5, 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sto_ng_single(ng, n, l, dtype):
    """
    Test normalization of all STO-NG basis functions
    """
    atol = 1.0e-6 if dtype == torch.float32 else 1.0e-8

    alpha, coeff = slater.to_gauss(ng, n, l, torch.tensor(1.0, dtype=dtype))
    angular = torch.tensor(l)
    vec = torch.zeros((3,), dtype=dtype)

    s = mmd.overlap((angular, angular), (alpha, alpha), (coeff, coeff), vec)
    ref = torch.diag(torch.ones((2 * l + 1,), dtype=dtype))

    assert torch.allclose(s, ref, atol=atol)
