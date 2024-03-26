"""
Testing dispersion gradient (autodiff).
"""

from __future__ import annotations

import pytest
import torch

from dxtb._types import DD
from dxtb.components.classicals.dispersion import new_dispersion
from dxtb.param import GFN1_XTB as par

from .samples import samples

sample_list = ["LiH", "SiH4", "MB16_43_01", "PbH4-BiH3"]

device = None


@pytest.mark.grad
@pytest.mark.parametrize("name", sample_list)
def test_grad_fail(name: str) -> None:
    dtype = torch.double
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    disp = new_dispersion(numbers, par, **dd)
    assert disp is not None

    cache = disp.get_cache(numbers)
    energy = disp.get_energy(positions, cache)

    with pytest.raises(RuntimeError):
        disp.get_gradient(energy, positions)
