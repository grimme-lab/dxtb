# This file is part of dxtb, modified from diffqc/dqc.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Original file licensed under the Apache License, Version 2.0 by diffqc/dqc.
# Modifications made by Grimme Group.
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
Symmetry: S1
============

S1 symmetry (no symmetry). In tight-binding, we do not require anything special here. Only S1 symmetry is of intereset.
"""
from __future__ import annotations

import numpy as np

from .base import BaseSymmetry

__all__ = ["S1Symmetry"]


class S1Symmetry(BaseSymmetry):
    """
    S1 Symmetry (no symmetry).
    """

    def get_reduced_shape(self, orig_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Get the reduced shape from the original shape.
        """
        return orig_shape

    @property
    def code(self) -> str:
        """
        Short code for this symmetry.
        """
        return "s1"

    def reconstruct_array(self, arr: np.ndarray, _: tuple[int, ...]) -> np.ndarray:
        """
        Reconstruct the full array from the reduced symmetrized array.
        """
        return arr
