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
Implementation: Multipole Base
==============================

Template for calculation and modification of multipole integrals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tad_mctc.batch import pack

from dxtb._src.typing import Tensor

from .base import IntegralLibcint

if TYPE_CHECKING:
    from .driver import IntDriverLibcint

__all__ = ["MultipoleLibcint"]


class MultipoleLibcint(IntegralLibcint):
    """
    Base class for multipole integrals calculated with `libcint`.
    """

    def multipole(self, driver: IntDriverLibcint, intstring: str) -> Tensor:
        """
        Calculation of multipole integral. The integral is normalized, using
        the diagonal of the overlap integral.

        Parameters
        ----------
        driver : IntDriver
            The integral driver for the calculation.
        intstring : str
            String for `libcint` integral engine.

        Returns
        -------
        Tensor
            Normalized multipole integral.
        """
        super().checks(driver)

        # pylint: disable=import-outside-toplevel
        from dxtb._src.exlibs import libcint

        allowed_mps = ("r0", "r0r0", "r0r0r0")
        if intstring not in allowed_mps:
            raise ValueError(
                f"Unknown integral string '{intstring}' provided.\n"
                f"Only '{', '.join(allowed_mps)} are allowed.\n'"
                "Other integrals can be added to `tad-libcint`."
            )

        def _mpint(driver: libcint.LibcintWrapper) -> Tensor:
            return libcint.int1e(intstring, driver)

        # batched mode
        if driver.ihelp.batch_mode > 0:
            if not isinstance(driver.drv, list):
                raise RuntimeError(
                    "IndexHelper on integral driver is batched, but the driver "
                    "instance itself not."
                )

            # In this version, batch mode does not matter. If we would
            # normalize the integral here, we would have to deflate the norm.
            self.matrix = pack([_mpint(driver) for driver in driver.drv])
            return self.matrix

        # single mode
        if not isinstance(driver.drv, libcint.LibcintWrapper):
            raise RuntimeError(
                "IndexHelper on integral driver is not batched, but the "
                "driver instance itself seems to be batched."
            )

        self.matrix = _mpint(driver.drv)
        return self.matrix
