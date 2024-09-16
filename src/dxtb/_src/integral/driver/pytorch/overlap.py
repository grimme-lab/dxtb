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

PyTorch-based overlap implementations.
"""

from __future__ import annotations

from tad_mctc.convert import symmetrize

from dxtb._src.typing import Tensor

from ...types import OverlapIntegral
from ...utils import snorm
from .base import IntegralPytorch
from .driver import BaseIntDriverPytorch
from .impls import OverlapFunction

__all__ = ["OverlapPytorch"]


class OverlapPytorch(OverlapIntegral, IntegralPytorch):
    """
    Overlap integral from atomic orbitals.

    Use the :meth:`.build` method to calculate the overlap integral. The
    returned matrix uses a custom autograd function to calculate the
    backward pass with the analytical gradient.
    For the full gradient, i.e., a matrix of shape ``(..., norb, norb, 3)``,
    the :meth:`.get_gradient` method should be used.
    """

    def build(self, driver: BaseIntDriverPytorch) -> Tensor:
        """
        Overlap calculation of unique shells pairs, using the
        McMurchie-Davidson algorithm.

        Parameters
        ----------
        driver : BaseIntDriverPytorch
            The integral driver for the calculation.

        Returns
        -------
        Tensor
            Overlap integral matrix of shape ``(..., norb, norb)``.
        """
        super().checks(driver)

        if driver.ihelp.batch_mode > 0:
            self.matrix = self._batch(driver.eval_ovlp, driver)
        else:
            self.matrix = self._single(driver.eval_ovlp, driver)

        # force symmetry to avoid problems through numerical errors
        if self.uplo == "n":
            return symmetrize(self.matrix, force=False)

        self.norm = snorm(self.matrix)
        return self.matrix

    def get_gradient(self, driver: BaseIntDriverPytorch) -> Tensor:
        """
        Overlap gradient calculation of unique shells pairs, using the
        McMurchie-Davidson algorithm.

        Parameters
        ----------
        driver : BaseIntDriverPytorch
            Integral driver for the calculation.

        Returns
        -------
        Tensor
            Overlap gradient of shape ``(..., norb, norb, 3)``.
        """
        super().checks(driver)

        # build norm if not already available
        if self.norm is None:
            self.build(driver)

        if driver.ihelp.batch_mode > 0:
            self.gradient = self._batch(driver.eval_ovlp_grad, driver)
        else:
            self.gradient = self._single(driver.eval_ovlp_grad, driver)

        return self.gradient

    def _single(
        self, fcn: OverlapFunction, driver: BaseIntDriverPytorch
    ) -> Tensor:
        if not isinstance(driver, BaseIntDriverPytorch):
            raise RuntimeError("Wrong integral driver selected.")

        return fcn(
            driver._positions_single,
            driver.basis,
            driver.ihelp,
            self.uplo,
            self.cutoff,
        )

    def _batch(
        self, fcn: OverlapFunction, driver: BaseIntDriverPytorch
    ) -> Tensor:
        if not isinstance(driver, BaseIntDriverPytorch):
            raise RuntimeError("Wrong integral driver selected.")

        # pylint: disable=import-outside-toplevel
        from tad_mctc.batch import pack

        return pack(
            [
                fcn(
                    driver._positions_batch[_batch],
                    driver._basis_batch[_batch],
                    driver._ihelp_batch[_batch],
                    self.uplo,
                    self.cutoff,
                )
                for _batch in range(driver.numbers.shape[0])
            ]
        )
