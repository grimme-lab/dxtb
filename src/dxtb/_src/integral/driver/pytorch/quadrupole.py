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
Implementation: Quadrupole
==========================

PyTorch-based quadrupole integral implementations.
"""

from __future__ import annotations

import torch

from dxtb._src.constants import defaults
from dxtb._src.typing import Literal, Tensor

from ...types import QuadrupoleIntegral
from .driver import IntDriverPytorch
from .impls.pipeline import QUADRUPOLE_COMPONENTS
from .multipole import MultipolePytorch

__all__ = ["QuadrupolePytorch"]


class QuadrupolePytorch(QuadrupoleIntegral, MultipolePytorch):
    """
    Quadrupole integral from atomic orbitals.
    """

    uplo: Literal["n", "u", "l"] = "l"
    """
    Whether the matrix of unique shell pairs should be create as a
    triangular matrix (``l``: lower, ``u``: upper) or full matrix (``n``).
    Defaults to ``l`` (lower triangular matrix).
    """

    cutoff: Tensor | float | int | None = defaults.INTCUTOFF
    """
    Real-space cutoff for integral calculation in Bohr. Defaults to
    ``constants.defaults.INTCUTOFF``.
    """

    def __init__(
        self,
        uplo: Literal["n", "N", "u", "U", "l", "L"] = "l",
        cutoff: Tensor | float | int | None = defaults.INTCUTOFF,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.cutoff = cutoff

        if uplo not in ("n", "N", "u", "U", "l", "L"):
            raise ValueError(f"Unknown option for `uplo` chosen: '{uplo}'.")
        self.uplo = uplo.casefold()  # type: ignore

    def build(self, driver: IntDriverPytorch) -> Tensor:
        """
        Raw (traced, 9-component, row-major) quadrupole integral about the
        Cartesian origin (``r0r0``), using the algorithm selected on the driver
        (``int_algorithm``). The reduction to six components, the shift and the
        traceless conversion are shared post-processing steps.

        Parameters
        ----------
        driver : IntDriverPytorch
            Integral driver for the calculation.

        Returns
        -------
        Tensor
            Integral of shape ``(..., 9, norb, norb)``.
        """
        return self.multipole(driver, QUADRUPOLE_COMPONENTS)

    def get_gradient(self, driver: IntDriverPytorch) -> Tensor:
        """
        Quadrupole intgral gradient calculation of unique shells pairs, using the
        McMurchie-Davidson algorithm.

        Parameters
        ----------
        driver : IntDriverPytorch
            Integral driver for the calculation.

        Returns
        -------
        Tensor
            Integral gradient of shape ``(..., norb, norb, 3, 3)``.
        """
        super().checks(driver)
        raise NotImplementedError
