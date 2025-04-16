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
General test for Core GFN1Hamiltonian.
"""
# pylint: disable=protected-access

from __future__ import annotations

import pytest
import torch
from tad_mctc.convert import str_to_device
from tad_mctc.typing import MockTensor

from dxtb import GFN1_XTB, GFN2_XTB, IndexHelper, Param, ParamModule
from dxtb._src.xtb.gfn1 import GFN1Hamiltonian
from dxtb._src.xtb.gfn2 import GFN2Hamiltonian


@pytest.mark.parametrize("par", [GFN1_XTB, GFN2_XTB])
def test_no_h0_fail(par: Param) -> None:
    """Fail due to missing hamiltonian in constructor."""
    dummy = torch.tensor([])
    _par = par.model_copy(deep=True)
    _par.hamiltonian = None

    with pytest.raises(RuntimeError):
        GFN1Hamiltonian(dummy, _par, dummy)  # type: ignore

    with pytest.raises(RuntimeError):
        GFN2Hamiltonian(dummy, _par, dummy)  # type: ignore


def test_no_h0_fail_2() -> None:
    """Fail due to missing hamiltonian in _get_hscale."""
    numbers = torch.tensor([1])
    par_gfn1 = GFN1_XTB.model_copy(deep=True)
    par_gfn2 = GFN2_XTB.model_copy(deep=True)

    ihelp_gfn1 = IndexHelper.from_numbers(numbers, par_gfn1)
    ihelp_gfn2 = IndexHelper.from_numbers(numbers, par_gfn2)

    h0_gfn1 = GFN1Hamiltonian(numbers, par_gfn1, ihelp_gfn1)
    par_gfn1.hamiltonian = None
    h0_gfn2 = GFN2Hamiltonian(numbers, par_gfn2, ihelp_gfn2)
    par_gfn2.hamiltonian = None

    with pytest.raises(RuntimeError):
        h0_gfn1._get_hscale(ParamModule(par_gfn1))

    with pytest.raises(RuntimeError):
        h0_gfn2._get_hscale(ParamModule(par_gfn2))


def test_no_h0_fail_3() -> None:
    """Fail due to missing shell information."""
    numbers = torch.tensor([1])
    par_gfn2 = GFN2_XTB.model_copy(deep=True)
    ihelp_gfn2 = IndexHelper.from_numbers(numbers, par_gfn2)

    assert par_gfn2.hamiltonian is not None
    par_gfn2.hamiltonian.xtb.shell = {}

    with pytest.raises(KeyError):
        GFN2Hamiltonian(numbers, par_gfn2, ihelp_gfn2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    """Change type of GFN1Hamiltonian."""
    numbers = torch.tensor([1])
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})
    h0 = GFN1Hamiltonian(numbers, GFN1_XTB, ihelp)
    assert h0.type(dtype).dtype == dtype


def test_change_type_fail() -> None:
    """Fail due to wrong type."""
    numbers = torch.tensor([1])
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})
    h0 = GFN1Hamiltonian(numbers, GFN1_XTB, ihelp)

    # trying to use setter
    with pytest.raises(AttributeError):
        h0.dtype = torch.float64

    # passing disallowed dtype
    with pytest.raises(ValueError):
        h0.type(torch.bool)


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    """Change device of GFN1Hamiltonian."""
    device = str_to_device(device_str)

    numbers = torch.tensor([1], device=device)
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})
    h0 = GFN1Hamiltonian(numbers, GFN1_XTB, ihelp, device=device)

    if device_str == "cpu":
        dev = torch.device("cpu")
    elif device_str == "cuda":
        dev = torch.device("cuda:0")
    else:
        assert False

    h0 = h0.to(dev)
    assert h0.device == dev


def test_change_device_fail() -> None:
    """Fail due to wrong device."""
    numbers = torch.tensor([1])
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})
    h0 = GFN1Hamiltonian(numbers, GFN1_XTB, ihelp)

    # trying to use setter
    with pytest.raises(AttributeError):
        h0.device = "cpu"


def test_wrong_device_fail() -> None:
    """Fail due to wrong device."""
    numbers_cpu = torch.tensor([1], device=torch.device("cpu"))
    ihelp = IndexHelper.from_numbers_angular(numbers_cpu, {1: [0]})

    numbers = MockTensor(torch.tensor([1], dtype=torch.float32))
    numbers.device = torch.device("cuda")

    # numbers is on a different device
    with pytest.raises(ValueError):
        GFN1Hamiltonian(numbers, GFN1_XTB, ihelp, device=torch.device("cpu"))
