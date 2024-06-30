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
Test the SCF guess.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import IndexHelper, labels
from dxtb._src.scf import guess

from ..conftest import DEVICE

numbers = torch.tensor([6, 1], device=DEVICE)
positions = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    device=DEVICE,
)
ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0, 0], 6: [0, 1]})
charge = torch.tensor(0.0, device=DEVICE)


def test_fail() -> None:
    with pytest.raises(ValueError):
        guess.get_guess(numbers, positions, charge, ihelp, name="eht")

    with pytest.raises(ValueError):
        guess.get_guess(numbers, positions, charge, ihelp, name=1000)

    # charges change because IndexHelper is broken
    with pytest.raises(RuntimeError):
        ih = IndexHelper.from_numbers_angular(numbers, {1: [0, 0], 6: [0, 1]})
        ih.orbitals_to_shell = torch.tensor([1, 2, 3])
        guess.get_guess(numbers, positions, charge, ih)


@pytest.mark.parametrize("name", ["eeq", labels.GUESS_EEQ])
def test_eeq(name: str | int) -> None:
    c = guess.get_guess(numbers, positions, charge, ihelp, name=name)
    ref = torch.tensor(
        [
            -0.11593066900969,
            -0.03864355757833,
            -0.03864355757833,
            -0.03864355757833,
            +0.11593066900969,
            +0.11593066900969,
        ]
    )

    assert pytest.approx(ref.cpu(), abs=1e-5) == c.cpu()


@pytest.mark.parametrize("name", ["sad", labels.GUESS_SAD])
def test_sad(name: str | int) -> None:
    c = guess.get_guess(numbers, positions, charge, ihelp, name=name)
    size = int(ihelp.orbitals_per_shell.sum().item())

    assert pytest.approx(torch.zeros(size).cpu()) == c.cpu()
