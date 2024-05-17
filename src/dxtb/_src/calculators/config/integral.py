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
Integral configuration.
"""

from __future__ import annotations

from dxtb._src.constants import defaults, labels
from dxtb._src.typing import Literal

__all__ = ["ConfigIntegrals"]


class ConfigIntegrals:
    """
    Configuration for the integrals.

    All configuration options are represented as integers. String options are
    converted to integers in the constructor.
    """

    level: int
    """
    Indicator for integrals to compute.

    - 0: None
    - 1: overlap
    - 2: +core Hamiltonian
    - 3: +dipole
    - 4: +quadrupole
    """

    cutoff: float
    """
    Real-space cutoff (in Bohr) for integral evaluation for PyTorch.
    The ``libint`` driver ignores this option.
    """

    driver: int
    """Type of integral driver."""

    uplo: Literal["n", "l", "u"]
    """Integral mode for PyTorch integral calculation."""

    def __init__(
        self,
        *,
        level: int = defaults.INTLEVEL,
        cutoff: float = defaults.INTCUTOFF,
        driver: str | int = defaults.INTDRIVER,
        uplo: str = defaults.INTUPLO,
    ) -> None:
        self.cutoff = cutoff

        if not isinstance(level, int):
            raise TypeError(
                f"The received integral level (`{level}`) is not an integer, "
                f"but {type(level)}."
            )
        self.level = level

        if uplo not in ("n", "N", "u", "U", "l", "L"):
            raise ValueError(f"Unknown option for `uplo` chosen: '{uplo}'.")
        self.uplo = uplo.casefold()  # type: ignore

        if isinstance(driver, str):
            if driver.casefold() in labels.INTDRIVER_LIBCINT_STRS:
                self.driver = labels.INTDRIVER_LIBCINT
            elif driver.casefold() in labels.INTDRIVER_ANALYTICAL_STRS:
                self.driver = labels.INTDRIVER_ANALYTICAL
            elif driver.casefold() in labels.INTDRIVER_AUTOGRAD_STRS:
                self.driver = labels.INTDRIVER_AUTOGRAD
            else:
                raise ValueError(f"Unknown integral driver '{driver}'.")
        elif isinstance(driver, int):
            if driver not in (
                labels.INTDRIVER_LIBCINT,
                labels.INTDRIVER_ANALYTICAL,
                labels.INTDRIVER_AUTOGRAD,
            ):
                raise ValueError(f"Unknown integral driver '{driver}'.")

            self.driver = driver
        else:
            raise TypeError(
                "The driver must be of type 'int' or 'str', but "
                f"'{type(driver)}' was given."
            )
