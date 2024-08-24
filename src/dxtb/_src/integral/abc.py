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
Integrals: Abstract Base Classes
================================

Abstract case class for integrals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from dxtb._src.typing import TYPE_CHECKING, Any, Tensor

if TYPE_CHECKING:
    from dxtb._src.integral.base import IntDriver


__all__ = ["IntegralABC"]


class IntegralABC(ABC):
    """
    Abstract base class for integral implementations.

    All integral calculations are executed by this class.
    """

    @abstractmethod
    def build(self, driver: IntDriver, **kwargs: Any) -> Tensor:
        """
        Create the integral matrix.

        Parameters
        ----------
        driver : IntDriver
            Integral driver for the calculation.

        Returns
        -------
        Tensor
            Integral matrix.
        """

    @abstractmethod
    def get_gradient(self, driver: IntDriver, **kwargs: Any) -> Tensor:
        """
        Calculate the full nuclear gradient matrix of the integral.

        Parameters
        ----------
        driver : IntDriver
            Integral driver for the calculation.

        Returns
        -------
        Tensor
            Nuclear integral derivative matrix.
        """

    @abstractmethod
    def normalize(self, norm: Tensor | None = None, **kwargs: Any) -> None:
        """
        Normalize the integral (changes ``self.matrix``).

        Parameters
        ----------
        norm : Tensor, optional
            Overlap norm to normalize the integral.
        """
