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
Implementation: Multipole (PyTorch)
===================================

Shared implementation of the PyTorch dipole and quadrupole integrals, built
from the interchangeable ``compute_1d`` kernels (see
:mod:`dxtb._src.integral.driver.pytorch.impls.kernels`).
"""

from __future__ import annotations

from tad_mctc.batch import pack

from dxtb._src.typing import Tensor

from .base import IntegralPytorch
from .driver import BaseIntDriverPytorch
from .impls.pairs import assemble_matrix

__all__ = ["MultipolePytorch"]


class MultipolePytorch(IntegralPytorch):
    """
    Base class for multipole integrals calculated with the PyTorch kernels.
    """

    def multipole(
        self,
        driver: BaseIntDriverPytorch,
        components: tuple[tuple[int, int, int], ...],
    ) -> Tensor:
        """
        Raw multipole integral about the Cartesian origin, i.e., the analogue
        of ``libcint``'s ``r0`` / ``r0r0`` (traced, unshifted) integrals.

        The shells are normalized to exactly unit self-overlap, so the
        integral needs no further normalization by the overlap diagonal
        (the overlap of every PyTorch driver has a unit diagonal, i.e., its
        norm is one).

        Parameters
        ----------
        driver : BaseIntDriverPytorch
            The integral driver for the calculation.
        components : tuple[tuple[int, int, int], ...]
            Per-axis multipole order of every output component.

        Returns
        -------
        Tensor
            Integral of shape ``(ncomp, nao, nao)`` (single) or
            ``(nbatch, ncomp, nao, nao)`` (batched).
        """
        super().checks(driver)

        kernel = driver.kernel

        def _one(ihelp, bas, pos) -> Tensor:
            alphas, coeffs = bas.create_cgtos()
            return assemble_matrix(
                kernel, ihelp, alphas, coeffs, pos, components
            )

        if driver.ihelp.batch_mode > 0:
            self.matrix = pack(
                [
                    _one(
                        driver._ihelp_batch[i],
                        driver._basis_batch[i],
                        driver._positions_batch[i],
                    )
                    for i in range(driver.numbers.shape[0])
                ]
            )
        else:
            self.matrix = _one(
                driver.ihelp, driver.basis, driver._positions_single
            )

        return self.matrix
