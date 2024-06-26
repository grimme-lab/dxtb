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
Testing SCF guess gradient (autodiff).
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.batch import pack

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.constants import labels
from dxtb._src.scf import guess
from dxtb._src.typing import DD, Callable, Tensor

from ..conftest import DEVICE
from .samples import samples

sample_list = ["H2", "HHe", "LiH", "H2O", "SiH4"]

tol = 1e-7


def gradchecker(
    dtype: torch.dtype, name: str, guess_name: int
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    ihelp = IndexHelper.from_numbers(numbers, par)

    # variables to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return guess.get_guess(numbers, pos, charge, ihelp, name=guess_name)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("guess_name", [labels.GUESS_EEQ, labels.GUESS_SAD])
def test_grad(dtype: torch.dtype, name: str, guess_name: int) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, guess_name)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("guess_name", [labels.GUESS_EEQ, labels.GUESS_SAD])
def test_gradgrad(dtype: torch.dtype, name: str, guess_name: int) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, guess_name)
    assert dgradgradcheck(func, diffvars, atol=tol)


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str, guess_name: int
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    charge = torch.tensor([0.0, 0.0], **dd)

    ihelp = IndexHelper.from_numbers(numbers, par)

    # variables to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return guess.get_guess(numbers, pos, charge, ihelp, name=guess_name)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("guess_name", [labels.GUESS_EEQ, labels.GUESS_SAD])
def test_grad_batch(
    dtype: torch.dtype, name1: str, name2: str, guess_name: int
) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2, guess_name)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("guess_name", [labels.GUESS_EEQ, labels.GUESS_SAD])
def test_gradgrad_batch(
    dtype: torch.dtype, name1: str, name2: str, guess_name: int
) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2, guess_name)
    assert dgradgradcheck(func, diffvars, atol=tol)
