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
Test culling (removing of systems) from IndexHelper.
"""

from __future__ import annotations

import torch

from dxtb._types import Slicers, Tensor
from dxtb.basis import IndexHelper
from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch


def test_culling() -> None:
    numbers = batch.pack(
        [
            torch.tensor([3, 1]),  # LiH
            torch.tensor([14, 1, 1, 1, 1]),  # SiH4
        ]
    )
    ihelp = IndexHelper.from_numbers(numbers, par)
    ref_ihelp = IndexHelper.from_numbers(numbers, par)

    conv = torch.tensor([False, True])
    slicers: Slicers = {
        "orbital": [slice(0, i) for i in [torch.tensor(6)]],
        "shell": [slice(0, i) for i in [torch.tensor(4)]],
        "atom": [slice(0, i) for i in [torch.tensor(2)]],
    }

    ihelp.cull(conv, slicers=slicers)

    for name in ihelp.__slots__:
        attr = getattr(ihelp, name)
        if isinstance(attr, Tensor):
            if name not in (
                "unique_angular",
                "ushells_to_unique",
                "ushells_per_unique",
            ):
                # get attribute from normal ihelp and remove padding
                a = getattr(ref_ihelp, name)[0]
                ref = batch.deflate(a, value=a[-1])

                assert (ref == attr.squeeze()).all()


def test_no_action() -> None:
    numbers = batch.pack(
        [
            torch.tensor([3, 1]),  # LiH
            torch.tensor([14, 1, 1, 1, 1]),  # SiH4
        ]
    )
    ihelp = IndexHelper.from_numbers(numbers, par)
    ref_ihelp = IndexHelper.from_numbers(numbers, par)

    conv = torch.tensor([False, True])
    slicers: Slicers = {
        "orbital": (...,),
        "shell": (...,),
        "atom": (...,),
    }

    ihelp.cull(conv, slicers=slicers)

    for name in ihelp.__slots__:
        attr = getattr(ihelp, name)
        if isinstance(attr, Tensor):
            if name not in (
                "unique_angular",
                "ushells_to_unique",
                "ushells_per_unique",
            ):
                ref = getattr(ref_ihelp, name)[0]
                assert (ref == attr.squeeze()).all()
