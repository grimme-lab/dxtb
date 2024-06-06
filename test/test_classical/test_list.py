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
Test collection of classical contributions `ClassicalList`.
"""

from __future__ import annotations

import torch

from dxtb import IndexHelper
from dxtb._src.components.classicals import ClassicalList, ClassicalListCache

from ..conftest import DEVICE


def test_empty() -> None:
    clist = ClassicalList()
    assert len(clist.components) == 0

    numbers = torch.tensor([6, 1], device=DEVICE)
    positions = torch.tensor([[0, 0, 0], [0, 0, 1]], device=DEVICE)
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0, 0], 6: [0, 1]})

    c = clist.get_cache(numbers, ihelp)
    assert isinstance(c, ClassicalListCache)

    e = clist.get_energy(positions, c)
    assert "none" in e
    assert (e["none"] == torch.zeros(e["none"].shape)).all()

    g = clist.get_gradient(e, positions)
    assert "none" in e
    assert (g["none"] == torch.zeros(g["none"].shape)).all()
