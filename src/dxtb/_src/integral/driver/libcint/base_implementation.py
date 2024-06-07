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
Libcint: Base Implementation
============================

Base class for `libcint`-based integral implementations.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch

from dxtb._src.constants import labels
from dxtb._src.typing import Self, Tensor, override

from ...base import BaseIntegralImplementation

if TYPE_CHECKING:
    from .driver import IntDriverLibcint
del TYPE_CHECKING


__all__ = ["IntegralImplementationLibcint", "LibcintImplementation"]


class LibcintImplementation:
    """
    Simple label for `libcint`-based integral implementations.
    """

    family: int = labels.INTDRIVER_LIBCINT
    """Label for integral implementation family"""

    def checks(self, driver: IntDriverLibcint) -> None:
        """
        Check if the type of integral driver is correct.

        Parameters
        ----------
        driver : IntDriverLibcint
            Integral driver for the calculation.
        """
        # pylint: disable=import-outside-toplevel
        from .driver import IntDriverLibcint

        if not isinstance(driver, IntDriverLibcint):
            raise RuntimeError("Wrong integral driver selected.")


class IntegralImplementationLibcint(
    LibcintImplementation,
    BaseIntegralImplementation,
):
    """PyTorch-based integral implementation"""

    def checks(self, driver: IntDriverLibcint) -> None:
        """
        Check if the type of integral driver is correct.

        Parameters
        ----------
        driver : BaseIntDriverPytorch
            Integral driver for the calculation.
        """
        super().checks(driver)

        # pylint: disable=import-outside-toplevel
        from .driver import IntDriverLibcint

        if not isinstance(driver, IntDriverLibcint):
            raise RuntimeError("Wrong integral driver selected.")

    def get_gradient(self, _: IntDriverLibcint) -> Tensor:
        """
        Create the nuclear integral derivative matrix.

        Parameters
        ----------
        driver : IntDriver
            Integral driver for the calculation.

        Returns
        -------
        Tensor
            Nuclear integral derivative matrix.
        """
        raise NotImplementedError(
            "The `get_gradient` method is not implemented for libcint "
            "integrals as it is not explicitly required."
        )

    @abstractmethod
    def build(self, driver: IntDriverLibcint) -> Tensor:
        """
        Calculation of the integral using libcint.

        Returns
        -------
        driver : IntDriverLibcint
            The integral driver for the calculation.
        """

    @override
    def to(self, device: torch.device) -> Self:
        """
        Returns a copy of the :class:`.IntegralImplementationLibcint` instance
        on the specified device.

        This method overwrites the usual approach because the
        :class:`.IntegralImplementationLibcint` class should not change the
        device of the norm .

        Parameters
        ----------
        device : torch.device
            Device to which all associated tensors should be moved.

        Returns
        -------
        BaseIntegral
            A copy of the :class:`.IntegralImplementationLibcint` instance
            placed on the specified device.

        Raises
        ------
        RuntimeError
            If the ``__slots__`` attribute is not set in the class.
        """
        if self.device == device:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `to` method requires setting ``__slots__`` in the "
                f"'{self.__class__.__name__}' class."
            )

        self.matrix = self.matrix.to(device)
        if self._gradient is not None:
            self.gradient = self.gradient.to(device)

        self.override_device(device)
        return self
