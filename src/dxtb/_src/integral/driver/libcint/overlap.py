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

from dxtb._src.exlibs import libcint
from dxtb._src.typing import Tensor

from .base_implementation import IntegralImplementationLibcint
from .driver import IntDriverLibcint

__all__ = ["OverlapLibcint"]


def snorm(ovlp: Tensor) -> Tensor:
    return torch.pow(ovlp.diagonal(dim1=-1, dim2=-2), -0.5)


class OverlapLibcint(IntegralImplementationLibcint):
    """
    Overlap integral from atomic orbitals.
    """

    def build(self, driver: IntDriverLibcint) -> Tensor:
        """
        Calculation of overlap integral using libcint.

        Returns
        -------
        driver : IntDriverLibcint
            The integral driver for the calculation.
        """
        super().checks(driver)

        def fcn(driver: libcint.LibcintWrapper) -> tuple[Tensor, Tensor]:
            s = libcint.overlap(driver)
            norm = snorm(s)

            if self.normalize is True:
                s = einsum("...ij,...i,...j->...ij", s, norm, norm)

            return s, norm

        # batched mode
        if driver.ihelp.batch_mode > 0:
            assert isinstance(driver.drv, list)

            slist = []
            nlist = []

            for d in driver.drv:
                mat, norm = fcn(d)
                slist.append(mat)
                nlist.append(norm)

            self.norm = pack(nlist)
            self.matrix = pack(slist)
            return self.matrix

        # single mode
        assert isinstance(driver.drv, libcint.LibcintWrapper)

        self.matrix, self.norm = fcn(driver.drv)
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
            Overlap gradient of shape `(nb, norb, norb, 3)`.
        """
        super().checks(driver)

        def fcn(driver: libcint.LibcintWrapper, norm: Tensor) -> Tensor:
            # (3, norb, norb)
            grad = libcint.int1e("ipovlp", driver)

            if self.normalize is False:
                return -einsum("...xij->...ijx", grad)

            # normalize and move xyz dimension to last, which is required for
            # the reduction (only works with extra dimension in last)
            return -einsum("...xij,...i,...j->...ijx", grad, norm, norm)

        # build norm if not already available
        if self.norm is None:
            if driver.ihelp.batch_mode > 0:
                assert isinstance(driver.drv, list)
                self.norm = pack([snorm(libcint.overlap(d)) for d in driver.drv])
            else:
                assert isinstance(driver.drv, libcint.LibcintWrapper)
                self.norm = snorm(libcint.overlap(driver.drv))

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

                self.grad = pack(
                    [
                        fcn(driver, deflate(norm))
                        for driver, norm in zip(driver.drv, self.norm)
                    ]
                )
                return self.grad
            elif driver.ihelp.batch_mode == 2:
                self.grad = pack(
                    [
                        fcn(driver, norm)  # no deflating here
                        for driver, norm in zip(driver.drv, self.norm)
                    ]
                )
                return self.grad

            raise ValueError(f"Unknown batch mode '{driver.ihelp.batch_mode}'.")

        # single mode
        if not isinstance(driver.drv, libcint.LibcintWrapper):
            raise RuntimeError(
                "IndexHelper on integral driver is not batched, but the "
                "driver instance itself seems to be batched."
            )

        self.grad = fcn(driver.drv, self.norm)
        return self.grad
