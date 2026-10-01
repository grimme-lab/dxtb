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
Run tests for the energy and potential of the spin-polarization interaction.

The shell-resolved magnetization charges are taken from tblite.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, GFN2_XTB, IndexHelper
from dxtb._src.components.interactions.spin import (
    ANGULAR_PAIR_INDEX,
    SpinPolarizationCache,
    load_spin_constants,
    new_spin_polarization,
)
from dxtb._src.typing import DD, Slicers, Tensor

from ..conftest import DEVICE
from .samples import samples

sample_list = [("H2+", "gfn1"), ("OH", "gfn1"), ("OH", "gfn2"), ("LiH", "gfn2")]


def reference_energy(
    numbers: Tensor, ihelp: IndexHelper, qsh: Tensor
) -> Tensor:
    """Spin energy from explicit loops as implemented in tblite."""
    table = load_spin_constants(dtype=qsh.dtype, device=qsh.device)
    angular = ihelp.angular.tolist()
    sh2at = ihelp.shells_to_atom.tolist()

    e = torch.tensor(0.0, dtype=qsh.dtype, device=qsh.device)
    for i, (li, ati) in enumerate(zip(angular, sh2at)):
        for j, (lj, atj) in enumerate(zip(angular, sh2at)):
            if ati != atj:
                continue
            w = table[numbers[ati], ANGULAR_PAIR_INDEX[li][lj]]
            e = e + 0.5 * qsh[i] * w * qsh[j]
    return e


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name,gfn", sample_list)
def test_single(dtype: torch.dtype, name: str, gfn: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps)
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    qsh = sample[f"qsh{gfn}"].to(**dd)[1]  # magnetization channel
    par = GFN1_XTB if gfn == "gfn1" else GFN2_XTB

    ihelp = IndexHelper.from_numbers(numbers, par)
    spin = new_spin_polarization(numbers, **dd)
    cache = spin.get_cache(numbers=numbers, ihelp=ihelp)

    e = spin.get_monopole_shell_energy(cache, qsh)
    ref = reference_energy(numbers, ihelp, qsh)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == e.sum(-1).cpu()

    # potential is the derivative of the energy
    v = spin.get_monopole_shell_potential(cache, qsh)
    assert pytest.approx((2 * e).cpu(), abs=tol, rel=tol) == (qsh * v).cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2+", "OH"])
@pytest.mark.parametrize("name2", ["H2+", "OH"])
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps)
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    qsh = pack(
        (
            sample1["qshgfn1"].to(**dd)[1],
            sample2["qshgfn1"].to(**dd)[1],
        )
    )

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    spin = new_spin_polarization(numbers, **dd)
    cache = spin.get_cache(numbers=numbers, ihelp=ihelp)
    e = spin.get_monopole_shell_energy(cache, qsh).sum(-1)

    ref = torch.stack(
        [
            reference_energy(
                s["numbers"].to(DEVICE),
                IndexHelper.from_numbers(s["numbers"].to(DEVICE), GFN1_XTB),
                s["qshgfn1"].to(**dd)[1],
            )
            for s in (sample1, sample2)
        ]
    )
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == e.cpu()


def test_matrix() -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}

    # carbon with s, p, and d shell, hydrogen with s shell
    numbers = torch.tensor([6, 1], device=DEVICE)
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0], 6: [0, 1, 2]})
    table = load_spin_constants(**dd)

    spin = new_spin_polarization(numbers, wscale=0.5, **dd)
    cache = spin.get_cache(numbers=numbers, ihelp=ihelp)

    ref = torch.zeros((4, 4), **dd)
    ref[:3, :3] = table[6, torch.tensor(ANGULAR_PAIR_INDEX, device=DEVICE)]
    ref[3, 3] = table[1, 0]

    # no interatomic blocks
    assert pytest.approx(0.5 * ref.cpu(), abs=1e-12) == cache.wll.cpu()


def test_cache() -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}

    numbers = torch.tensor([6, 1], device=DEVICE)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    spin = new_spin_polarization(numbers, **dd)

    cache = spin.get_cache(numbers=numbers, ihelp=ihelp)
    assert spin.get_cache(numbers=numbers, ihelp=ihelp) is cache

    # different basis for the same atoms
    ihelp2 = IndexHelper.from_numbers(numbers, GFN2_XTB)
    cache2 = spin.get_cache(numbers=numbers, ihelp=ihelp2)
    assert cache2 is not cache
    assert cache2.wll.shape == (ihelp2.nsh, ihelp2.nsh)

    # updated scaling
    spin.update(wscale=torch.tensor(0.5, **dd))
    cache3 = spin.get_cache(numbers=numbers, ihelp=ihelp2)
    assert pytest.approx((0.5 * cache2.wll).cpu()) == cache3.wll.cpu()


def test_cull() -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}

    nums = [torch.tensor([3, 1], device=DEVICE), samples["OH"]["numbers"]]
    numbers = pack(nums)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    spin = new_spin_polarization(numbers, **dd)
    cache = spin.get_cache(numbers=numbers, ihelp=ihelp)
    assert isinstance(cache, SpinPolarizationCache)
    wll = cache.wll.clone()

    with pytest.raises(RuntimeError):
        cache.restore()

    nsh = IndexHelper.from_numbers(nums[0], GFN1_XTB).nsh
    slicers: Slicers = {
        "orbital": (...,),
        "shell": [slice(0, nsh)],
        "atom": [slice(0, 2)],
    }
    cache.cull(torch.tensor([False, True], device=DEVICE), slicers)
    assert pytest.approx(wll[:1, :nsh, :nsh].cpu()) == cache.wll.cpu()

    cache.restore()
    assert pytest.approx(wll.cpu()) == cache.wll.cpu()
