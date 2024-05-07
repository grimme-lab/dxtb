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
General test for Core Hamiltonian.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.convert import str_to_device

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.xtb.gfn1 import GFN1Hamiltonian as Hamiltonian


def test_no_h0_fail() -> None:
    dummy = torch.tensor([])
    _par = par.model_copy(deep=True)
    _par.hamiltonian = None

    with pytest.raises(RuntimeError):
        Hamiltonian(dummy, _par, dummy)  # type: ignore


def test_no_h0_fail2() -> None:
    numbers = torch.tensor([1])
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})
    _par = par.model_copy(deep=True)
    h0 = Hamiltonian(numbers, _par, ihelp)

    _par.hamiltonian = None
    with pytest.raises(RuntimeError):
        h0._get_hscale()  # pylint: disable=protected-access

    with pytest.raises(RuntimeError):
        h0.build(numbers, numbers)

    with pytest.raises(RuntimeError):
        h0.get_gradient(
            numbers,
            numbers,
            numbers,
            numbers,
            numbers,
            numbers,  # type: ignore
            numbers,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    numbers = torch.tensor([1])
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})
    h0 = Hamiltonian(numbers, par, ihelp)
    assert h0.type(dtype).dtype == dtype


def test_change_type_fail() -> None:
    numbers = torch.tensor([1])
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})
    h0 = Hamiltonian(numbers, par, ihelp)

    # trying to use setter
    with pytest.raises(AttributeError):
        h0.dtype = torch.float64

    # passing disallowed dtype
    with pytest.raises(ValueError):
        h0.type(torch.bool)


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    device = str_to_device(device_str)

    numbers = torch.tensor([1], device=device)
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})
    h0 = Hamiltonian(numbers, par, ihelp, device=device)

    if device_str == "cpu":
        dev = torch.device("cpu")
    elif device_str == "cuda":
        dev = torch.device("cuda:0")

    h0 = h0.to(dev)
    assert h0.device == dev


def test_change_device_fail() -> None:
    numbers = torch.tensor([1])
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})
    h0 = Hamiltonian(numbers, par, ihelp)

    # trying to use setter
    with pytest.raises(AttributeError):
        h0.device = "cpu"


@pytest.mark.cuda
def test_wrong_device_fail() -> None:
    numbers = torch.tensor([1], device=str_to_device("cuda"))
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})

    # numbers is on a different device
    with pytest.raises(ValueError):
        Hamiltonian(numbers, par, ihelp)
