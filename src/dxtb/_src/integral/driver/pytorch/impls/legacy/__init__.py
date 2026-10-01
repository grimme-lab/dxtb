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
Legacy overlap
==============

The original loop-based overlap matrix build (``overlap``) with explicitly
written-down McMurchie-Davidson E-coefficients (``explicit``). It is kept for
reference and used only by the legacy driver (``int_driver="legacy"``); all
other integrals are built by the pair builder (``impls/pairs.py``).
"""

from .explicit import md_explicit, md_explicit_gradient
from .overlap import overlap_gradient_legacy, overlap_legacy

__all__ = [
    "md_explicit",
    "md_explicit_gradient",
    "overlap_legacy",
    "overlap_gradient_legacy",
]
