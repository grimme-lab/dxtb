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
General test for base Hamiltonian.
"""

from __future__ import annotations

import tempfile as td
from pathlib import Path

import pytest
import torch
from tad_mctc._version import __tversion__

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.xtb.gfn1 import GFN1Hamiltonian


def test_requires_grad() -> None:
    numbers = torch.tensor([1])
    ihelp = IndexHelper.from_numbers(numbers, par)

    h = GFN1Hamiltonian(numbers, par, ihelp)

    h._matrix = None
    assert h.requires_grad is False

    h._matrix = torch.tensor([1.0], requires_grad=True)
    assert h.requires_grad is True


def test_write_to_pt() -> None:
    numbers = torch.tensor([3, 1])
    ihelp = IndexHelper.from_numbers(numbers, par)

    h = GFN1Hamiltonian(numbers, par, ihelp)
    h._matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    kwargs: dict = {"map_location": torch.device("cpu")}
    if __tversion__ > (1, 12, 1):
        kwargs["weights_only"] = True

    with td.TemporaryDirectory() as tmpdir:
        p_write = Path(tmpdir) / "test.pt"
        h.to_pt(p_write)

        read_mat = torch.load(p_write, **kwargs)
        assert pytest.approx(h._matrix.cpu()) == read_mat

    with td.TemporaryDirectory() as tmpdir:
        p_write = Path(tmpdir) / f"{h.label.casefold()}"

        # To test the None case, inject the temporary path via the label
        h.label = str(p_write)
        h.to_pt()

        read_mat = torch.load(f"{p_write}.pt", **kwargs)
        assert pytest.approx(h._matrix.cpu()) == read_mat
