import pytest
import torch

from xtbml.basis import orthogonalize, slater
from xtbml.integral import mmd


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_ortho_1s_2s(dtype):
    """Test orthogonality of 1s and 2s orbitals"""

    # azimuthal quantum number of s-orbital
    l = torch.tensor(0)

    # same site
    vec = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)

    # create gaussians
    alphai, coeffi = slater.to_gauss(
        torch.tensor(5),
        torch.tensor(1),
        l,
        vec.new_tensor(1.2),
    )
    alphaj, coeffj = slater.to_gauss(
        torch.tensor(2),
        torch.tensor(2),
        l,
        vec.new_tensor(0.7),
    )

    alphaj, coeffj = orthogonalize((alphai, alphaj), (coeffi, coeffj))

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
