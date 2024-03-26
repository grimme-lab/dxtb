"""
Run tests for repulsion contribution.

(Note that the analytical gradient tests fail for `torch.float`.)
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb._types import DD, Literal
from dxtb.basis import IndexHelper
from dxtb.components.classicals import new_repulsion
from dxtb.param.gfn1 import GFN1_XTB
from dxtb.param.gfn2 import GFN2_XTB
from dxtb.utils import batch

from .samples import samples

sample_list = ["H2", "H2O", "SiH4", "ZnOOH-", "MB16_43_01", "MB16_43_02", "LYS_xao"]

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("par", ["gfn1", "gfn2"])
def test_single(dtype: torch.dtype, name: str, par: Literal["gfn1", "gfn2"]) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]

    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = sample[par].to(**dd)

    if par == "gfn1":
        _par = GFN1_XTB
    elif par == "gfn2":
        _par = GFN2_XTB
    else:
        assert False

    rep = new_repulsion(numbers, _par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, _par)
    cache = rep.get_cache(numbers, ihelp)
    e = rep.get_energy(positions, cache, atom_resolved=False)

    assert pytest.approx(ref, abs=tol) == 0.5 * e.sum((-2, -1))


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("par", ["gfn1", "gfn2"])
def test_batch(
    dtype: torch.dtype, name1: str, name2: str, par: Literal["gfn1", "gfn2"]
) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    positions = batch.pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    ref = torch.stack(
        [
            sample1[par].to(**dd),
            sample2[par].to(**dd),
        ],
    )

    if par == "gfn1":
        _par = GFN1_XTB
    elif par == "gfn2":
        _par = GFN2_XTB
    else:
        assert False

    rep = new_repulsion(numbers, _par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, _par)
    cache = rep.get_cache(numbers, ihelp)
    e = rep.get_energy(positions, cache)

    assert pytest.approx(ref, abs=tol) == e.sum(-1)
