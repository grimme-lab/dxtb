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
Implementation: Overlap
=======================

PyTorch-based overlap implementations.
"""

from __future__ import annotations

from tad_mctc.convert import symmetrize

from dxtb._src.typing import Tensor

from ...types import OverlapIntegral
from ...utils import snorm
from .base import IntegralPytorch
from .driver import IntDriverPytorch

__all__ = ["OverlapPytorch"]


class OverlapPytorch(OverlapIntegral, IntegralPytorch):
    """
    Overlap integral from atomic orbitals.

    Use the :meth:`.build` method to calculate the overlap integral, which is
    differentiable with autograd. There is no analytical overlap gradient in
    the PyTorch driver; the analytical calculator needs the libcint driver.
    """

    def build(self, driver: IntDriverPytorch) -> Tensor:
        """
        Overlap calculation with the driver's overlap function.

        Parameters
        ----------
        driver : IntDriverPytorch
            The integral driver for the calculation.

        Returns
        -------
        Tensor
            Overlap integral matrix of shape ``(..., norb, norb)``.
        """
        super().checks(driver)

        self.matrix = driver.eval_matrix()[..., 0, :, :]

        # force symmetry to avoid problems through numerical errors
        if self.uplo == "n":
            return symmetrize(self.matrix, force=False)

        self.norm = snorm(self.matrix)
        return self.matrix

    def get_gradient(self, driver: IntDriverPytorch) -> Tensor:
        """
        Not available: differentiate :meth:`.build` with autograd, or use the
        libcint driver for the analytical overlap gradient.
        """
        super().checks(driver)
        raise NotImplementedError(
            "The PyTorch integral driver has no analytical overlap gradient. "
            "Use autograd on the overlap or the libcint driver."
        )
