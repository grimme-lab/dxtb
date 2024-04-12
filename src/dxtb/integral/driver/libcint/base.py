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
Base class for Multipole Integrals
==================================

Calculation and modification of multipole integrals.
"""

from __future__ import annotations

import torch

from dxtb.constants import labels
from dxtb.exlibs import libcint
from dxtb.typing import TYPE_CHECKING, Tensor
from dxtb.utils.batch import pack

from ...base import BaseIntegralImplementation

if TYPE_CHECKING:
    from .driver import IntDriverLibcint


__all__ = ["MultipoleLibcint"]


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


class MultipoleLibcint(IntegralImplementationLibcint):
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

        allowed_mps = ("r0", "r0r0", "r0r0r0")
        if intstring not in allowed_mps:
            raise ValueError(
                f"Unknown integral string '{intstring}' provided.\n"
                f"Only '{', '.join(allowed_mps)} are allowed.\n'"
                "Other integrals can be added to `tad-libcint`."
            )

        if self.norm is None:
            raise RuntimeError("Norm must be set before building.")

        def _mpint(driver: libcint.LibcintWrapper, norm: Tensor) -> Tensor:
            return torch.einsum(
                "...ij,i,j->...ij", libcint.int1e(intstring, driver), norm, norm
            )

        # batched mode
        if driver.ihelp.batch_mode > 0:
            if not isinstance(driver.drv, list):
                raise RuntimeError(
                    "IndexHelper on integral driver is batched, but the driver "
                    "instance itself not."
                )
            if driver.ihelp.batch_mode == 1:
                # pylint: disable=import-outside-toplevel
                from tad_mctc.batch import deflate

                self.matrix = pack(
                    [
                        _mpint(driver, deflate(norm))
                        for driver, norm in zip(driver.drv, self.norm)
                    ]
                )
                return self.matrix
            elif driver.ihelp.batch_mode == 2:
                self.matrix = pack(
                    [
                        _mpint(driver, norm)  # no deflating here
                        for driver, norm in zip(driver.drv, self.norm)
                    ]
                )
                return self.matrix

            raise ValueError(f"Unknown batch mode '{driver.ihelp.batch_mode}'.")

        # single mode
        if not isinstance(driver.drv, libcint.LibcintWrapper):
            raise RuntimeError(
                "IndexHelper on integral driver is not batched, but the "
                "driver instance itself seems to be batched."
            )

        self.matrix = _mpint(driver.drv, self.norm)
        return self.matrix
