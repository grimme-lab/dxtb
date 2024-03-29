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
Integral Symmetry
=================

In tight-binding, we do not require anything special here. Only S1 symmetry is
of intereset.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np


class BaseSymmetry:
    """
    Base class for integral symmetry.
    """

    @abstractmethod
    def get_reduced_shape(self, orig_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Get the reduced shape from the original shape.
        """
        pass

    @property
    @abstractmethod
    def code(self) -> str:
        """
        Short code for this symmetry.
        """

    @abstractmethod
    def reconstruct_array(
        self, arr: np.ndarray, orig_shape: tuple[int, ...]
    ) -> np.ndarray:
        """
        Reconstruct the full array from the reduced symmetrized array.
        """


class S1Symmetry(BaseSymmetry):
    """
    S1 Symmetry (no symmetry).
    """

    def get_reduced_shape(self, orig_shape: tuple[int, ...]) -> tuple[int, ...]:
        return orig_shape

    @property
    def code(self) -> str:
        return "s1"

    def reconstruct_array(self, arr: np.ndarray, _: tuple[int, ...]) -> np.ndarray:
        return arr
