"""
Test orthogonality of GFN1-xTB's H1s and H2s orbitals.
"""

from __future__ import annotations

import pytest
import torch

from dxtb._types import DD
from dxtb.basis import slater_to_gauss
from dxtb.basis.ortho import gaussian_integral, orthogonalize
from dxtb.integral.driver.pytorch.impls.md import overlap_gto

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_ortho_1s_2s(dtype: torch.dtype):
    """Test orthogonality of GFN1-xTB's H1s and H2s orbitals"""
    tols = {"abs": 1e-6, "rel": 1e-6, "nan_ok": False}
    dd: DD = {"device": device, "dtype": dtype}

    # azimuthal quantum number of s-orbital
    l = torch.tensor(0)

    # same site
    vec = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)

    # create gaussians
    alphai, coeffi = slater_to_gauss(
        torch.tensor(5), torch.tensor(1), l, vec.new_tensor(1.2)
    )
    alphaj, coeffj = slater_to_gauss(
        torch.tensor(2), torch.tensor(2), l, vec.new_tensor(0.7)
    )

    alphaj_new, coeffj_new = orthogonalize((alphai, alphaj), (coeffi, coeffj))

    # normalised self-overlap
    ref = torch.tensor(1, **dd)
    s = overlap_gto((l, l), (alphaj, alphaj), (coeffj, coeffj), vec)
    assert pytest.approx(ref, **tols) == s.sum()
    s2 = gaussian_integral(*(alphaj, alphaj), *(coeffj, coeffj))
    assert pytest.approx(ref, **tols) == s2

    # orthogonal overlap
    ref = torch.tensor(0, **dd)
    s = overlap_gto((l, l), (alphai, alphaj_new), (coeffi, coeffj_new), vec)
    s2 = gaussian_integral(*(alphai, alphaj_new), *(coeffi, coeffj_new))
    assert pytest.approx(ref, **tols) == s.sum()
    assert pytest.approx(ref, **tols) == s2
