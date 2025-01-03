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

from dxtb import GFN1_XTB, IndexHelper
from dxtb.components.base import InteractionList, InteractionListCache
from dxtb.components.coulomb import new_es3

from ..conftest import DEVICE


def test_properties() -> None:
    numbers = torch.tensor([6, 1], device=DEVICE)
    es3 = new_es3(numbers, GFN1_XTB)
    ilist = InteractionList(es3)

    assert len(ilist.components) == 1
    assert len(ilist) == 1
    assert id(ilist.get_interaction("ES3")) == id(es3)


def test_empty() -> None:
    ilist = InteractionList()
    assert len(ilist.components) == 0

    numbers = torch.tensor([6, 1], device=DEVICE)
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0, 0], 6: [0, 1]})
    orbital = ihelp.spread_atom_to_orbital(numbers)

    c = ilist.get_cache(numbers, numbers, ihelp)
    assert isinstance(c, InteractionListCache)

    e = ilist.get_energy(orbital, numbers, ihelp)  # type: ignore
    assert (e == torch.zeros(e.shape, device=DEVICE)).all()

    g = ilist.get_gradient(e, e, e, ihelp)  # type: ignore
    assert (g == torch.zeros(g.shape, device=DEVICE)).all()
