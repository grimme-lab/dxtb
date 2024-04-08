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
Symmetry: S4
============

Four-fold symmetry: (...ijkl) == (...jikl) == (...ijlk) == (...jilk)
"""
from __future__ import annotations

import numpy as np
from tad_libcint.api import CSYMM

from ..utils import int2ctypes, np2ctypes
from .base import BaseSymmetry


class S4Symmetry(BaseSymmetry):
    """
    S4 Symmetry: (...ijkl) == (...jikl) == (...ijlk) == (...jilk)
    """

    def get_reduced_shape(self, orig_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Get the reduced shape from the original shape.
        """
        # the returned shape would be (..., i(j+1)/2, k(l+1)/2)
        self.__check_orig_shape(orig_shape)

        batchshape = orig_shape[:-4]
        ijshape = orig_shape[-4] * (orig_shape[-3] + 1) // 2
        klshape = orig_shape[-2] * (orig_shape[-1] + 1) // 2
        return (*batchshape, ijshape, klshape)

    @property
    def code(self) -> str:
        """
        Short code for this symmetry.
        """
        return "s4"

    def reconstruct_array(
        self, arr: np.ndarray, orig_shape: tuple[int, ...]
    ) -> np.ndarray:
        """
        Reconstruct the full array from the reduced symmetrized array.
        """
        # reconstruct the full array
        # arr: (..., ij/2, kl/2)
        self.__check_orig_shape(orig_shape)

        out = np.zeros(orig_shape, dtype=arr.dtype)
        fcn = CSYMM().fills4
        fcn(
            np2ctypes(out),
            np2ctypes(arr),
            int2ctypes(orig_shape[-4]),
            int2ctypes(orig_shape[-2]),
        )
        return out

    def __check_orig_shape(self, orig_shape: tuple[int, ...]):
        assert len(orig_shape) >= 4
        assert orig_shape[-4] == orig_shape[-3]
        assert orig_shape[-2] == orig_shape[-1]
