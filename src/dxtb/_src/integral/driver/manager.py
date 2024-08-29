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
Integrals: Driver Manager
=========================

The driver manager contains the selection logic, i.e., the it instantiates
the appropriate driver based on the configuration.
"""
from __future__ import annotations

import logging

import torch

from dxtb import IndexHelper, labels
from dxtb._src.param import Param
from dxtb._src.typing import TYPE_CHECKING, Any, Tensor, TensorLike

if TYPE_CHECKING:
    from ..base import IntDriver


__all__ = ["DriverManager"]


logger = logging.getLogger(__name__)


class DriverManager(TensorLike):
    """
    This class instantiates the appropriate driver based on the
    configuration passed to it.
    """

    __slots__ = ["_driver", "driver_type", "force_cpu_for_libcint"]

    def __init__(
        self,
        driver_type: int,
        _driver: IntDriver | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, dtype=dtype)

        # per default, libcint is run on the CPU
        self.force_cpu_for_libcint = kwargs.pop(
            "force_cpu_for_libcint",
            True if driver_type == labels.INTDRIVER_LIBCINT else False,
        )

        self.driver_type = driver_type
        self._driver = _driver

    @property
    def driver(self) -> IntDriver:
        if self._driver is None:
            raise RuntimeError(
                "No driver has been created yet. Run `create_driver` first."
            )

        return self._driver

    @driver.setter
    def driver(self, driver: IntDriver) -> None:
        self._driver = driver

    def create_driver(
        self, numbers: Tensor, par: Param, ihelp: IndexHelper
    ) -> None:
        if self.driver_type == labels.INTDRIVER_LIBCINT:
            # pylint: disable=import-outside-toplevel
            from .libcint import IntDriverLibcint as _IntDriver

            if self.force_cpu_for_libcint is True:
                device = torch.device("cpu")
                numbers = numbers.to(device=device)
                ihelp = ihelp.to(device=device)

        elif self.driver_type == labels.INTDRIVER_ANALYTICAL:
            # pylint: disable=import-outside-toplevel
            from .pytorch import IntDriverPytorch as _IntDriver

        elif self.driver_type == labels.INTDRIVER_AUTOGRAD:
            # pylint: disable=import-outside-toplevel
            from .pytorch import IntDriverPytorchNoAnalytical as _IntDriver

        else:
            raise ValueError(f"Unknown integral driver '{self.driver_type}'.")

        self.driver = _IntDriver(
            numbers, par, ihelp, device=ihelp.device, dtype=self.dtype
        )

    def setup_driver(self, positions: Tensor, **kwargs: Any) -> None:
        """
        Setup the integral driver (if not already done).

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        """
        logger.debug("Integral Driver: Start setup.")

        if self.force_cpu_for_libcint is True:
            positions = positions.to(device=torch.device("cpu"))

        if self.driver.is_latest(positions) is True:
            logger.debug("Integral Driver: Skip setup. Already done.")
            return

        self.driver.setup(positions, **kwargs)
        logger.debug("Integral Driver: Finished setup.")

    def invalidate_driver(self) -> None:
        """Invalidate the integral driver to require new setup."""
        self.driver.invalidate()
