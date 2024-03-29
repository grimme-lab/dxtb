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
Test InteractionList.
"""

from __future__ import annotations

import torch

from dxtb.basis import IndexHelper
from dxtb.components.interactions import InteractionList


def test_empty() -> None:
    ilist = InteractionList()
    assert len(ilist.components) == 0

    numbers = torch.tensor([6, 1])
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0, 0], 6: [0, 1]})
    orbital = ihelp.spread_atom_to_orbital(numbers)

    c = ilist.get_cache(numbers, numbers, ihelp)
    assert isinstance(c, InteractionList.Cache)

    e = ilist.get_energy(orbital, numbers, ihelp)  # type: ignore
    assert (e == torch.zeros(e.shape)).all()

    g = ilist.get_gradient(e, e, e, ihelp)  # type: ignore
    assert (g == torch.zeros(g.shape)).all()
