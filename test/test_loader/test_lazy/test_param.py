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
Test the lazy loaders.
"""

from __future__ import annotations

from dxtb._src.loader.lazy import LazyLoaderParam
import pytest
import pytest
from pathlib import Path


def test_lazy_loader_param_initialization() -> None:
    filepath = "test.toml"
    loader = LazyLoaderParam(filepath)
    assert loader.filepath == filepath
    assert loader._loaded is None


def test_lazy_loader_param_str() -> None:
    filepath = "test.toml"
    loader = LazyLoaderParam(filepath)
    assert str(loader) == f"LazyLoaderParam({filepath})"


def test_lazy_loader_param_repr() -> None:
    filepath = "test.toml"
    loader = LazyLoaderParam(filepath)
    assert repr(loader) == f"LazyLoaderParam({filepath})"


@pytest.mark.parametrize("parname", ["gfn1-xtb", "gfn2-xtb"])
def test_lazy_loader_param_equality(parname: str) -> None:
    p = (
        Path(__file__).parents[3]
        / "src"
        / "dxtb"
        / "_src"
        / "param"
        / parname.split("-")[0]
        / f"{parname}.toml"
    )

    loader1 = LazyLoaderParam(p)
    loader2 = LazyLoaderParam(p)

    # Trigger the lazy loading
    _ = loader1.meta
    _ = loader2.meta

    assert loader1 == loader2
    assert loader1._loaded == loader2._loaded
