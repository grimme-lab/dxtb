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
Implementation: Multipole (PyTorch)
===================================

Shared implementation of the PyTorch dipole and quadrupole integrals, built
from the interchangeable ``compute_1d`` kernels (see
:mod:`dxtb._src.integral.driver.pytorch.impls.kernels`).
"""

from __future__ import annotations

import torch

from dxtb._src.constants import defaults
from dxtb._src.typing import Literal, Tensor

from .base import IntegralPytorch
from .driver import IntDriverPytorch

__all__ = ["MultipolePytorch"]


class MultipolePytorch(IntegralPytorch):
    """
    Base class for multipole integrals calculated with the PyTorch kernels.
    """

    def __init__(
        self,
        uplo: Literal["n", "N", "u", "U", "l", "L"] = "l",
        cutoff: Tensor | float | int | None = defaults.INTCUTOFF,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device=device, dtype=dtype, uplo=uplo, cutoff=cutoff)

    def get_gradient(self, driver: IntDriverPytorch) -> Tensor:
        """
        Gradient of the multipole integral. Not implemented for the PyTorch
        driver; use autograd on the integral instead.
        """
        super().checks(driver)
        raise NotImplementedError

    def multipole(
        self,
        driver: IntDriverPytorch,
        components: tuple[tuple[int, int, int], ...],
    ) -> Tensor:
        """
        Raw multipole integral about the Cartesian origin, i.e., the analogue
        of ``libcint``'s ``r0`` / ``r0r0`` (traced, unshifted) integrals.

        The shells are normalized to exactly unit self-overlap, so the
        integral needs no further normalization by the overlap diagonal
        (the PyTorch overlap has a unit diagonal, i.e., its norm is one).

        Parameters
        ----------
        driver : IntDriverPytorch
            The integral driver for the calculation.
        components : tuple[tuple[int, int, int], ...]
            Per-axis multipole order of every output component.

        Returns
        -------
        Tensor
            Integral of shape ``(ncomp, nao, nao)`` (single) or
            ``(nbatch, ncomp, nao, nao)`` (batched).
        """
        super().checks(driver)

        self.matrix = driver.eval_matrix(components)
        return self.matrix
