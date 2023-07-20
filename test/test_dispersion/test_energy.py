"""
Testing dispersion energy.

These tests are taken from https://github.com/dftd3/tad-dftd3/tree/main/tests
and are only included for the sake of completeness.
"""
from __future__ import annotations

import pytest
import tad_dftd3 as d3
import torch

from dxtb._types import DD
from dxtb.dispersion import new_dispersion
from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch

from .samples import samples

device = None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_disp_batch(dtype: torch.dtype) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = (
        samples["PbH4-BiH3"],
        samples["C6H5I-CH3SH"],
    )
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
    c6 = batch.pack(
        (
            sample1["c6"].to(**dd),
            sample2["c6"].to(**dd),
        )
    )
    ref = batch.pack(
        (
            sample1["edisp"].to(**dd),
            sample2["edisp"].to(**dd),
        )
    )

    rvdw = d3.data.vdw_rad_d3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
    r4r2 = d3.data.sqrt_z_r4_over_r2[numbers]
    param = {
        "a1": torch.tensor(0.49484001, **dd),
        "s8": torch.tensor(0.78981345, **dd),
        "a2": torch.tensor(5.73083694, **dd),
    }

    energy = d3.disp.dispersion(
        numbers, positions, param, c6, rvdw, r4r2, d3.disp.rational_damping
    )
    assert energy.dtype == dtype
    assert torch.allclose(energy, ref)

    # create copy as `par` lives in global scope
    _par = par.model_copy(deep=True)
    if _par.dispersion is None or _par.dispersion.d3 is None:
        assert False

    # set parameters explicitly
    _par.dispersion.d3.a1 = param["a1"]
    _par.dispersion.d3.a2 = param["a2"]
    _par.dispersion.d3.s8 = param["s8"]

    disp = new_dispersion(numbers, _par, **dd)
    if disp is None:
        assert False

    cache = disp.get_cache(numbers)
    edisp = disp.get_energy(positions, cache)
    assert edisp.dtype == dtype
    assert pytest.approx(edisp) == ref
    assert pytest.approx(edisp) == energy
