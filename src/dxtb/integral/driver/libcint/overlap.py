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
Overlap implementation based on `libcint`.
"""

from __future__ import annotations

import torch
from tad_mctc.batch import deflate, pack
from tad_mctc.math import einsum

from dxtb._types import Tensor

from ...base import BaseIntegralImplementation
from .base import LibcintImplementation
from .driver import IntDriverLibcint
from .impls import LibcintWrapper, int1e, overlap

__all__ = ["OverlapLibcint"]


def snorm(ovlp: Tensor) -> Tensor:
    return torch.pow(ovlp.diagonal(dim1=-1, dim2=-2), -0.5)


class OverlapLibcint(BaseIntegralImplementation, LibcintImplementation):
    """
    Overlap integral from atomic orbitals.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)
        self._gradient = None

    @property
    def gradient(self) -> Tensor:
        if self._gradient is None:
            raise RuntimeError("Overlap gradient has not been calculated.")
        return self._gradient

    @gradient.setter
    def gradient(self, mat: Tensor) -> None:
        self._gradient = mat

    def build(self, driver: IntDriverLibcint) -> Tensor:
        """
        Calculation of overlap integral using libcint.

        Returns
        -------
        driver : IntDriverLibcint
            The integral driver for the calculation.
        """
        super().checks(driver)

        def fcn(driver: LibcintWrapper) -> tuple[Tensor, Tensor]:
            s = overlap(driver)
            norm = snorm(s)
            mat = einsum("...ij,...i,...j->...ij", s, norm, norm)
            return mat, norm

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
        assert isinstance(driver.drv, LibcintWrapper)

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

        def fcn(driver: LibcintWrapper, norm: Tensor) -> Tensor:
            # (3, norb, norb)
            grad = int1e("ipovlp", driver)

            # normalize and move xyz dimension to last, which is required for
            # the reduction (only works with extra dimension in last)
            return -einsum("...xij,...i,...j->...ijx", grad, norm, norm)

        # build norm if not already available
        if self.norm is None:
            if driver.ihelp.batch_mode > 0:
                assert isinstance(driver.drv, list)
                self.norm = pack([snorm(overlap(d)) for d in driver.drv])
            else:
                assert isinstance(driver.drv, LibcintWrapper)
                self.norm = snorm(overlap(driver.drv))

        # batched mode
        if driver.ihelp.batch_mode > 0:
            assert isinstance(driver.drv, list)

            if driver.ihelp.batch_mode == 1:
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
        assert isinstance(driver.drv, LibcintWrapper)

        self.grad = fcn(driver.drv, self.norm)
        return self.grad
