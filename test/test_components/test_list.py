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
Test collection of list of components.
"""

from __future__ import annotations

import pytest
import torch

from dxtb._src.components.list import ComponentList, ComponentListCache
from dxtb._src.typing import Any

from ..conftest import DEVICE


def test_cache() -> None:
    cache = ComponentListCache()

    dummy = torch.tensor([1, 2, 3], device=DEVICE)
    assert len(list(cache.keys())) == 0

    # dummy functions
    cache.cull(dummy, dummy)  # type: ignore
    cache.restore()


def test_list() -> None:
    class Dummy(ComponentList):
        def get_energy(self, *_: Any, **__: Any) -> None:
            pass

        def get_gradient(self, *_: Any, **__: Any) -> None:
            pass

        def get_cache(self, *_: Any, **__: Any) -> None:
            pass

    clist = Dummy()
    assert len(clist) == 0

    with pytest.raises(ValueError):
        clist.get_interaction("dummy")

    with pytest.raises(ValueError):
        clist.update("dummy")

    with pytest.raises(ValueError):
        clist.reset("dummy")
