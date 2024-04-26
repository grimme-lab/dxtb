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
Parametrization: GFN1-xTB Loader
================================

This module loads the standard GFN1-xTB parametrization (lazily).
Hence, the TOML file is only read when a member is accessed.

Example
-------
>>> from dxtb.param.gfn1 import GFN1_XTB
>>> print(GFN1_XTB._loaded is None)
True
>>> m = GFN1_XTB.meta
>>> print(GFN1_XTB._loaded is None)
False
"""

from __future__ import annotations

from pathlib import Path

from dxtb.loader.lazy import LazyLoaderParam as Lazy

from ..base import Param

GFN1_XTB: Param = Lazy(Path(__file__).parent / "gfn1-xtb.toml")  # type: ignore
