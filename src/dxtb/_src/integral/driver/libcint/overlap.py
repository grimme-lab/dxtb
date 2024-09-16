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
Implementation: Overlap
=======================

Overlap implementation based on `libcint`.
"""

from __future__ import annotations

import torch
from tad_mctc.batch import pack
from tad_mctc.math import einsum

from dxtb._src.typing import Tensor

from ...types import OverlapIntegral
from ...utils import snorm
from .base import IntegralLibcint
from .driver import IntDriverLibcint

__all__ = ["OverlapLibcint"]


class OverlapLibcint(OverlapIntegral, IntegralLibcint):
    """
    Overlap integral from atomic orbitals.

    Use the :meth:`build` method to calculate the overlap integral. The
    returned matrix uses a custom autograd function to calculate the
    backward pass with the analytical gradient.
    For the full gradient, i.e., a matrix of shape ``(..., norb, norb, 3)``,
    the :meth:`get_gradient` method should be used.
    """

    def build(self, driver: IntDriverLibcint) -> Tensor:
        """
        Calculation of overlap integral using libcint.

        Returns
        -------
        driver : IntDriverLibcint
            The integral driver for the calculation.

        Returns
        -------
        Tensor
            Overlap integral matrix of shape ``(..., norb, norb)``.
        """
        super().checks(driver)

        # pylint: disable=import-outside-toplevel
        from dxtb._src.exlibs import libcint

        # batched mode
        if driver.ihelp.batch_mode > 0:
            assert isinstance(driver.drv, list)

            slist = [libcint.overlap(d) for d in driver.drv]
            nlist = [snorm(s) for s in slist]

            self.norm = pack(nlist)
            self.matrix = pack(slist)
            return self.matrix

        # single mode
        assert isinstance(driver.drv, libcint.LibcintWrapper)

        self.matrix = libcint.overlap(driver.drv)
        self.norm = snorm(self.matrix)
        return self.matrix

    def get_gradient(self, driver: IntDriverLibcint) -> Tensor:
        """
        Overlap gradient calculation using libcint.

        Parameters
        ----------
        driver : IntDriverLibcint
            The integral driver for the calculation.

        Returns
        -------
        Tensor
            Overlap gradient of shape ``(..., norb, norb, 3)``.
        """
        super().checks(driver)

        # pylint: disable=import-outside-toplevel
        from dxtb._src.exlibs import libcint

        # build norm if not already available
        if self.norm is None:
            self.build(driver)

        def fcn(driver: libcint.LibcintWrapper) -> Tensor:
            # (3, norb, norb)
            grad = libcint.int1e("ipovlp", driver)

            # Move xyz dimension to last, which is required for the
            # reduction (only works with extra dimension in last)
            return -einsum("...xij->...ijx", grad)

        # batched mode
        if driver.ihelp.batch_mode > 0:
            if not isinstance(driver.drv, list):
                raise RuntimeError(
                    "IndexHelper on integral driver is batched, but the driver "
                    "instance itself not."
                )

            if driver.ihelp.batch_mode == 1:
                self.gradient = pack([fcn(d) for d in driver.drv])
                return self.gradient

            elif driver.ihelp.batch_mode == 2:
                self.gradient = torch.stack([fcn(d) for d in driver.drv])
                return self.gradient

            raise ValueError(f"Unknown batch mode '{driver.ihelp.batch_mode}'.")

        # single mode
        if not isinstance(driver.drv, libcint.LibcintWrapper):
            raise RuntimeError(
                "IndexHelper on integral driver is not batched, but the "
                "driver instance itself seems to be batched."
            )

        print("aksdjkasd")
        self.gradient = fcn(driver.drv)
        return self.gradient
