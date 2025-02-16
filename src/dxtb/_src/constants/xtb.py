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
`xtb` parameters
================

This module contains `xtb` parameters of the model that are not contained
in the parametrization file. Furthermore, `xtb`'s default values are stored here.
"""

# Electrostatics

DEFAULT_ES2_GEXP: float = 2.0
"""Default exponent of the second-order Coulomb interaction (2.0)."""


# Classical contributions

DEFAULT_XB_CUTOFF: float = 20.0
"""Default real space cutoff for halogen bonding interactions (20.0)."""

DEFAULT_REPULSION_CUTOFF: float = 25.0
"""Default real space cutoff for repulsion interactions."""
