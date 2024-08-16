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
from dxtb._src.constants import labels
from dxtb._src.param import Param
from dxtb._src.typing import TYPE_CHECKING, Any, Tensor

if TYPE_CHECKING:
    from ..base import IntDriver


__all__ = ["DriverManager"]


logger = logging.getLogger(__name__)


class DriverManager:
    """
    This class instantiates the appropriate driver based on the
    configuration passed to it.
    """

    def __init__(
        self,
        driver_type: int,
        numbers: Tensor,
        par: Param,
        ihelp: IndexHelper,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        self.device = device
        self.dtype = dtype

        # per default, libcint is run on the CPU
        self.force_cpu_for_libcint = kwargs.pop(
            "force_cpu_for_libcint",
            True if driver_type == labels.INTDRIVER_LIBCINT else False,
        )

        self.driver = self._select_driver(driver_type, numbers, par, ihelp)
        self.driver_type = driver_type

    def _select_driver(
        self, driver_type: int, numbers: Tensor, par: Param, ihelp: IndexHelper
    ) -> IntDriver:

        if driver_type == labels.INTDRIVER_LIBCINT:
            from .libcint import IntDriverLibcint

            if self.force_cpu_for_libcint:
                device = torch.device("cpu")
                numbers = numbers.to(device=device)
                ihelp = ihelp.to(device=device)

            return IntDriverLibcint(
                numbers, par, ihelp, device=self.device, dtype=self.dtype
            )

        elif driver_type == labels.INTDRIVER_ANALYTICAL:
            from .pytorch import IntDriverPytorch

            return IntDriverPytorch(
                numbers, par, ihelp, device=self.device, dtype=self.dtype
            )

        elif driver_type == labels.INTDRIVER_AUTOGRAD:
            from .pytorch import IntDriverPytorchNoAnalytical

            return IntDriverPytorchNoAnalytical(
                numbers, par, ihelp, device=self.device, dtype=self.dtype
            )

        else:
            raise ValueError(f"Unknown integral driver '{driver_type}'.")

    def setup_driver(self, positions: Tensor, **kwargs: Any) -> None:
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
