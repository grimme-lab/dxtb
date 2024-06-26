# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Testing dispersion energy.

These tests are taken from https://github.com/dftd3/tad-dftd3/tree/main/tests
and are only included for the sake of completeness.
"""

from __future__ import annotations

import pytest
import tad_dftd3 as d3
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB as par
from dxtb._src.components.classicals.dispersion import new_dispersion
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_disp_batch(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample1, sample2 = (
        samples["PbH4-BiH3"],
        samples["C6H5I-CH3SH"],
    )
    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    c6 = pack(
        (
            sample1["c6"].to(**dd),
            sample2["c6"].to(**dd),
        )
    )
    ref = pack(
        (
            sample1["edisp"].to(**dd),
            sample2["edisp"].to(**dd),
        )
    )

    rvdw = d3.data.VDW_D3.to(**dd)[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
    r4r2 = d3.data.R4R2.to(**dd)[numbers]
    param = {
        "a1": torch.tensor(0.49484001, **dd),
        "s8": torch.tensor(0.78981345, **dd),
        "a2": torch.tensor(5.73083694, **dd),
    }

    energy = d3.disp.dispersion(
        numbers, positions, param, c6, rvdw, r4r2, d3.disp.rational_damping
    )
    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu()) == energy.cpu()

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
    assert pytest.approx(edisp.cpu()) == ref.cpu()
    assert pytest.approx(edisp.cpu()) == energy.cpu()
