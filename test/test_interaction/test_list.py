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

from dxtb import GFN1_XTB, GFN2_XTB, IndexHelper
from dxtb.components.base import InteractionList, InteractionListCache
from dxtb.components.coulomb import new_es2, new_es3
from dxtb.components.dispersion import new_d4sc
from dxtb.components.field import new_efield, new_efield_grad

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


def test_reset() -> None:
    numbers = torch.tensor([6, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=DEVICE)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    d4sc = new_d4sc(numbers, GFN2_XTB, device=DEVICE)
    es2 = new_es2(numbers, GFN1_XTB, device=DEVICE)
    es3 = new_es3(numbers, GFN1_XTB, device=DEVICE)
    ef = new_efield(torch.ones((3,), device=DEVICE), device=DEVICE)
    efg = new_efield_grad(torch.ones((3, 3), device=DEVICE), device=DEVICE)

    ilist = InteractionList(d4sc, es2, es3, ef, efg)
    assert len(ilist.components) == 5

    _ = ilist.get_cache(numbers=numbers, positions=positions, ihelp=ihelp)

    assert d4sc is not None and d4sc.cache is not None
    assert es2 is not None and es2.cache is not None
    assert es3 is not None and es3.cache is not None
    assert ef is not None and ef.cache is not None
    assert efg is not None and efg.cache is not None

    assert len(d4sc.cache) == 3
    ilist.reset_d4sc()
    assert d4sc.cache is None

    assert len(es2.cache) == 2  # mat + shell_resolved
    ilist.reset_es2()
    assert es2.cache is None

    assert len(es3.cache) == 2  # hd + shell_resolved
    ilist.reset_es3()
    assert es3.cache is None

    assert len(ef.cache) == 2
    ilist.reset_efield()
    assert ef.cache is None

    assert len(efg.cache) == 1
    ilist.reset_efield_grad()
    assert efg.cache is None


def test_reset_all() -> None:
    numbers = torch.tensor([6, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=DEVICE)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    d4sc = new_d4sc(numbers, GFN2_XTB, device=DEVICE)
    es2 = new_es2(numbers, GFN1_XTB, device=DEVICE)
    es3 = new_es3(numbers, GFN1_XTB, device=DEVICE)
    ef = new_efield(torch.ones((3,), device=DEVICE), device=DEVICE)
    efg = new_efield_grad(torch.ones((3, 3), device=DEVICE), device=DEVICE)

    ilist = InteractionList(d4sc, es2, es3, ef, efg)
    assert len(ilist.components) == 5

    _ = ilist.get_cache(numbers=numbers, positions=positions, ihelp=ihelp)

    assert d4sc is not None and d4sc.cache is not None
    assert es2 is not None and es2.cache is not None
    assert es3 is not None and es3.cache is not None
    assert ef is not None and ef.cache is not None
    assert efg is not None and efg.cache is not None

    assert len(d4sc.cache) == 3
    assert len(es2.cache) == 2
    assert len(es3.cache) == 2
    assert len(ef.cache) == 2
    assert len(efg.cache) == 1

    ilist.reset_all()
    assert d4sc.cache is None
    assert es2.cache is None
    assert es3.cache is None
    assert ef.cache is None
    assert efg.cache is None


def test_update() -> None:
    numbers = torch.tensor([6, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=DEVICE)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    d4sc = new_d4sc(numbers, GFN2_XTB, device=DEVICE)
    es2 = new_es2(numbers, GFN1_XTB, device=DEVICE)
    es3 = new_es3(numbers, GFN1_XTB, device=DEVICE)
    ef = new_efield(torch.ones((3,), device=DEVICE), device=DEVICE)
    efg = new_efield_grad(torch.ones((3, 3), device=DEVICE), device=DEVICE)

    ilist = InteractionList(d4sc, es2, es3, ef, efg)
    assert len(ilist.components) == 5

    _ = ilist.get_cache(numbers=numbers, positions=positions, ihelp=ihelp)

    assert d4sc is not None and d4sc.cache is not None
    assert es2 is not None and es2.cache is not None
    assert es3 is not None and es3.cache is not None
    assert ef is not None and ef.cache is not None
    assert efg is not None and efg.cache is not None

    r4r2 = torch.tensor([1.0, 0.0, 0.0], device=DEVICE)
    ilist.update_d4sc(r4r2=r4r2)
    assert (d4sc.r4r2 == r4r2).all()

    lhubbard = torch.tensor([1.0, 0.0, 0.0], device=DEVICE)
    ilist.update_es2(lhubbard=lhubbard)
    assert (es2.lhubbard == lhubbard).all()

    hubbard_derivs = torch.tensor([1.0, 0.0, 0.0], device=DEVICE)
    ilist.update_es3(hubbard_derivs=hubbard_derivs)
    assert (es3.hubbard_derivs == hubbard_derivs).all()

    field = torch.tensor([1.0, 0.0, 0.0], device=DEVICE)
    ilist.update_efield(field=field)
    assert (ef.field == field).all()

    field_grad = torch.ones((3, 3), device=DEVICE)
    ilist.update_efield_grad(field_grad=field_grad)
    assert (efg.field_grad == field_grad).all()
