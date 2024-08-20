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
from .base import IntegralPytorch
from .driver import BaseIntDriverPytorch

__all__ = ["QuadrupolePytorch"]


class QuadrupolePytorch(QuadrupoleIntegral, IntegralPytorch):
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

        raise NotImplementedError(
            "PyTorch versions of multipole moments are not implemented. "
            "Use `libcint` as integral driver."
        )

    def build(self, driver: BaseIntDriverPytorch) -> Tensor:
        """
        Integral calculation of unique shells pairs, using the
        McMurchie-Davidson algorithm.

        Parameters
        ----------
        driver : BaseIntDriverPytorch
            Integral driver for the calculation.

        Returns
        -------
        Tensor
            Integral matrix of shape ``(..., norb, norb, 3)``.
        """
        super().checks(driver)
        raise NotImplementedError

    def get_gradient(self, driver: BaseIntDriverPytorch) -> Tensor:
        """
        Quadrupole intgral gradient calculation of unique shells pairs, using the
        McMurchie-Davidson algorithm.

        Parameters
        ----------
        driver : BaseIntDriverPytorch
            Integral driver for the calculation.

        Returns
        -------
        Tensor
            Integral gradient of shape ``(..., norb, norb, 3, 3)``.
        """
        super().checks(driver)
        raise NotImplementedError
