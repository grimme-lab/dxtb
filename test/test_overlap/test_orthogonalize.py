import pytest
import torch

from xtbml.basis import slater, orthogonalize
from xtbml.integral import mmd


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ortho_1s_2s(dtype):
    """Test orthogonality of 1s and 2s orbitals"""

    # azimuthal quantum number of s-orbital
    l = 0

    # same site
    vec = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)

    # create gaussians
    alphai, coeffi = slater.to_gauss(5, 1, l, torch.tensor(1.2, dtype=dtype))
    alphaj, coeffj = slater.to_gauss(2, 2, l, torch.tensor(0.7, dtype=dtype))

    alphaj, coeffj = orthogonalize(l, (alphai, alphaj), (coeffi, coeffj))

    # normalised self-overlap
    s = mmd.overlap((l, l), (alphaj, alphaj), (coeffj, coeffj), vec)
    assert torch.allclose(
        s, torch.eye(1, dtype=dtype), rtol=1e-05, atol=1e-05, equal_nan=False
    )

    # orthogonal overlap
    s = mmd.overlap((l, l), (alphai, alphaj), (coeffi, coeffj), vec)
    assert torch.allclose(
        s, torch.zeros(1, dtype=dtype), rtol=1e-05, atol=1e-05, equal_nan=False
    )
