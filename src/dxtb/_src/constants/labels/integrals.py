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
Labels: Integrals
=================

All labels related to integrals and their computation.
"""

# integral driver
INTDRIVER_LIBCINT = 0
"""Integer code for LIBCINT driver."""

INTDRIVER_LIBCINT_STRS = ("libcint", "c")
"""String codes for LIBCINT driver."""

INTDRIVER_AUTOGRAD = 1
"""Integer code for Autograd driver."""

INTDRIVER_AUTOGRAD_STRS = ("autograd", "pytorch", "torch", "dxtb")
"""String codes for Autograd driver."""

INTDRIVER_ANALYTICAL = 2
"""Integer code for Analytical driver."""

INTDRIVER_ANALYTICAL_STRS = ("analytical", "pytorch2", "torch2", "dxtb2")
"""String codes for Analytical driver."""

INTDRIVER_LEGACY = 3
"""Integer code for Legacy driver."""

INTDRIVER_LEGACY_STRS = ("legacy", "old", "loop")
"""String codes for Legacy driver."""

INTDRIVER_MAP = ["libcint", "Autograd", "Analytical", "Legacy (loops)"]
"""String map (for printing) of integral drivers."""

# levels

INTLEVEL_NONE = 0
"""No integrals."""

INTLEVEL_OVERLAP = 1
"""Overlap integrals."""

INTLEVEL_DIPOLE = 2
"""Dipole integrals."""

INTLEVEL_QUADRUPOLE = 3
"""Quadrupole integrals."""
