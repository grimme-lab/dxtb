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
PyTorch: Base Implementation
============================

Base class for PyTorch-based integral implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dxtb._src.constants import labels

from ...base import BaseIntegralImplementation

if TYPE_CHECKING:
    from .base_driver import BaseIntDriverPytorch
del TYPE_CHECKING

__all__ = ["IntegralImplementationPytorch"]


class PytorchImplementation:
    """
    Simple label for `PyTorch`-based integral implementations.
    """

    family: int = labels.INTDRIVER_ANALYTICAL
    """Label for integral implementation family"""


class IntegralImplementationPytorch(
    PytorchImplementation,
    BaseIntegralImplementation,
):
    """PyTorch-based integral implementation"""

    def checks(self, driver: BaseIntDriverPytorch) -> None:
        """
        Check if the type of integral driver is correct.

        Parameters
        ----------
        driver : BaseIntDriverPytorch
            Integral driver for the calculation.
        """
        super().checks(driver)

        # pylint: disable=import-outside-toplevel
        from .base_driver import BaseIntDriverPytorch

        if not isinstance(driver, BaseIntDriverPytorch):
            raise RuntimeError("Wrong integral driver selected.")
