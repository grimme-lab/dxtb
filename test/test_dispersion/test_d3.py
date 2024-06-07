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
Testing dispersion energy and autodiff.

These tests are taken from https://github.com/awvwgk/tad-dftd3/tree/main/tests
and are only included for the sake of completeness.
"""

from __future__ import annotations

import pytest
import tad_dftd3 as d3
import torch

from dxtb import GFN1_XTB as par
from dxtb._src.components.classicals.dispersion import DispersionD3, new_dispersion
from dxtb._src.typing import DD, Tensor
from dxtb._src.utils import batch

from ..conftest import DEVICE
from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_disp_batch(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample1, sample2 = (
        samples["PbH4-BiH3"],
        samples["C6H5I-CH3SH"],
    )
    numbers = batch.pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
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


@pytest.mark.grad
def test_grad_pos() -> None:
    dtype = torch.double
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples["C4H5NCS"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd).detach().clone()

    # variable to be differentiated
    positions.requires_grad_(True)

    disp = new_dispersion(numbers, par, **dd)
    if disp is None:
        assert False

    cache = disp.get_cache(numbers)

    def func(positions: Tensor) -> Tensor:
        return disp.get_energy(positions, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, positions)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
def test_grad_pos_tblite(dtype: torch.dtype) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples["PbH4-BiH3"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd).detach().clone()
    positions.requires_grad_(True)
    ref = sample["grad"].to(**dd)

    disp = new_dispersion(numbers, par, **dd)
    if disp is None:
        assert False

    cache = disp.get_cache(numbers)

    # automatic gradient
    energy = torch.sum(disp.get_energy(positions, cache), dim=-1)
    energy.backward()

    if positions.grad is None:
        assert False
    grad_backward = positions.grad.clone()

    assert pytest.approx(grad_backward.cpu(), abs=1e-10) == ref.cpu()


@pytest.mark.grad
def test_grad_param() -> None:
    dtype = torch.double
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples["C4H5NCS"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    param = (
        torch.tensor(1.00000000, requires_grad=True, dtype=dtype),
        torch.tensor(0.78981345, requires_grad=True, dtype=dtype),
        torch.tensor(0.49484001, requires_grad=True, dtype=dtype),
        torch.tensor(5.73083694, requires_grad=True, dtype=dtype),
    )
    label = ("s6", "s8", "a1", "a2")

    def func(*inputs):
        input_param = {label[i]: input for i, input in enumerate(inputs)}
        disp = DispersionD3(numbers, input_param, **dd)
        cache = disp.get_cache(numbers)
        return disp.get_energy(positions, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, param)
