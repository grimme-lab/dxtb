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

These tests are taken from https://github.com/dftd4/tad-dftd4/tree/main/test
and are only included for the sake of completeness.
"""

from __future__ import annotations

import pytest
import tad_dftd4 as d4
import torch

from dxtb.components.classicals.dispersion import DispersionD4, new_dispersion
from dxtb.param.gfn2 import GFN2_XTB as par
from dxtb.typing import DD, Tensor
from dxtb.utils import batch

from .samples import samples

device = None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch(dtype: torch.dtype) -> None:
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
    charge = positions.new_zeros(numbers.shape[0])

    param = {
        "a1": positions.new_tensor(0.49484001),
        "a2": positions.new_tensor(5.73083694),
        "s8": positions.new_tensor(0.78981345),
        "s9": positions.new_tensor(1.00000000),
    }

    energy = d4.dftd4(numbers, positions, charge, param)
    assert energy.dtype == dtype

    # create copy as `par` lives in global scope
    _par = par.model_copy(deep=True)
    if _par.dispersion is None or _par.dispersion.d4 is None:
        assert False

    # set parameters explicitly
    _par.dispersion.d4.a1 = param["a1"]
    _par.dispersion.d4.a2 = param["a2"]
    _par.dispersion.d4.s8 = param["s8"]
    _par.dispersion.d4.s9 = param["s9"]

    disp = new_dispersion(numbers, _par, charge=charge, **dd)
    if disp is None:
        assert False

    cache = disp.get_cache(numbers)
    edisp = disp.get_energy(positions, cache)
    assert edisp.dtype == dtype
    assert pytest.approx(edisp) == energy


@pytest.mark.grad
def test_grad_pos() -> None:
    dtype = torch.double
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples["C4H5NCS"]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd).detach()
    charge = positions.new_tensor(0.0)

    # variable to be differentiated
    positions.requires_grad_(True)

    disp = new_dispersion(numbers, par, charge, **dd)
    if disp is None:
        assert False

    cache = disp.get_cache(numbers)

    def func(positions: Tensor) -> Tensor:
        return disp.get_energy(positions, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, positions)


@pytest.mark.grad
def test_grad_param() -> None:
    dtype = torch.double
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples["C4H5NCS"]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    charge = positions.new_tensor(0.0)
    param = (
        positions.new_tensor(1.00000000, requires_grad=True),
        positions.new_tensor(0.78981345, requires_grad=True),
        positions.new_tensor(0.49484001, requires_grad=True),
        positions.new_tensor(5.73083694, requires_grad=True),
    )
    label = ("s6", "s8", "a1", "a2")

    def func(*inputs):
        input_param = {label[i]: input for i, input in enumerate(inputs)}
        disp = DispersionD4(numbers, input_param, charge, **dd)
        cache = disp.get_cache(numbers)
        return disp.get_energy(positions, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, param)
