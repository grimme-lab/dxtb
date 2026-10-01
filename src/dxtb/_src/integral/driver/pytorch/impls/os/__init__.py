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
Obara-Saika algorithm
======================

The three-index Obara-Saika vertical recursion for the McMurchie-
Davidson-compatible ``compute_1d`` contract, for the overlap (``emax == 0``)
and multipole moments about a common origin (``emax`` 1 and 2).
"""

from .compute_1d import compute_1d_os

__all__ = ["compute_1d_os"]
