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
Labels: Dictionary Keys
=======================

All labels related to special dictionary keys.
"""

__all__ = [
    "KEY_NUMBERS",
    "KEY_POSITIONS",
    "KEY_CHARGE",
    "KEY_SPIN",
    #
    "KEY_ENERGY",
    "KEY_FORCES",
    "KEY_GRADIENT",
    "KEY_DIPOLE",
    #
    "KEY_REF_ENERGY",
    "KEY_REF_FORCES",
    "KEY_REF_GRADIENT",
    "KEY_REF_DIPOLE",
]

KEY_NUMBERS = "numbers"
KEY_POSITIONS = "positions"
KEY_CHARGE = "charge"
KEY_SPIN = "spin"

KEY_ENERGY = "energy"
KEY_FORCES = "forces"
KEY_GRADIENT = "gradient"
KEY_DIPOLE = "dipole"

KEY_REF_ENERGY = "ref_energy"
KEY_REF_FORCES = "ref_forces"
KEY_REF_GRADIENT = "ref_gradient"
KEY_REF_DIPOLE = "ref_dipole"
