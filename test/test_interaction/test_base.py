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
Test Interaction.
"""

from __future__ import annotations

import torch

from dxtb import IndexHelper
from dxtb._src.components.base import ComponentCache
from dxtb._src.components.interactions import Charges, Interaction
from ..conftest import DEVICE


def test_empty() -> None:
    i = Interaction()
    assert i.label == "Interaction"

    numbers = torch.tensor([6, 1], device=DEVICE)
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0, 0], 6: [0, 1]})
    orbital = ihelp.spread_atom_to_orbital(numbers)

    c = i.get_cache(numbers=numbers, positions=numbers, ihelp=ihelp)
    assert isinstance(c, ComponentCache)

    ae = i.get_atom_energy(numbers)
    assert (ae == torch.zeros(ae.shape, device=DEVICE)).all()

    se = i.get_shell_energy(numbers)
    assert (se == torch.zeros(se.shape, device=DEVICE)).all()

    e = i.get_energy(Charges(mono=orbital), numbers, ihelp)  # type: ignore
    assert (e == torch.zeros(e.shape, device=DEVICE)).all()

    ap = i.get_atom_potential(numbers)
    assert (ap == torch.zeros(ap.shape, device=DEVICE)).all()

    sp = i.get_shell_potential(numbers, numbers)
    assert (sp == torch.zeros(sp.shape, device=DEVICE)).all()

    ag = i.get_atom_gradient(numbers, numbers)
    assert (ag == torch.zeros(ag.shape, device=DEVICE)).all()

    sg = i.get_shell_gradient(numbers, numbers)
    assert (sg == torch.zeros(sg.shape, device=DEVICE)).all()
