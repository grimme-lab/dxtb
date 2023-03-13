"""
Testing dispersion energy.

These tests are taken from https://github.com/dftd3/tad-dftd3/tree/main/tests
and are only included for the sake of completeness.
"""
from __future__ import annotations

import pytest
import tad_dftd3 as d3
import torch

from dxtb.dispersion import new_dispersion
from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch

from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_disp_batch(dtype: torch.dtype) -> None:
    dd = {"dtype": dtype}

    sample1, sample2 = (
        samples["PbH4-BiH3"],
        samples["C6H5I-CH3SH"],
    )
    numbers = batch.pack(
        (
            sample1["numbers"],
            sample2["numbers"],
        )
    )
    positions = batch.pack(
        (
            sample1["positions"].type(dtype),
            sample2["positions"].type(dtype),
        )
    )
    c6 = batch.pack(
        (
            sample1["c6"].type(dtype),
            sample2["c6"].type(dtype),
        )
    )
    ref = batch.pack(
        (
            sample1["edisp"].type(dtype),
            sample2["edisp"].type(dtype),
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
    _par = par.copy(deep=True)
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
