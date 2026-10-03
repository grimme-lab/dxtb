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
Driver: PyTorch
===============

Collection of PyTorch-based integral drivers.
"""

from __future__ import annotations

import torch
from tad_mctc.batch import deflate, pack

from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.typing import Any, Tensor

from ...base import IntDriver
from .base import PytorchImplementation
from .impls.kernels import DEFAULT_ALGORITHM, get_kernel
from .impls.pairs import PairPlan, assemble_matrix, prepare
from .impls.pipeline import Kernel1D

__all__ = ["IntDriverPytorch"]


class IntDriverPytorch(PytorchImplementation, IntDriver):
    """
    PyTorch-based integral driver.

    All integrals are built by the pair builder with the 1D kernel selected
    in :attr:`algorithm` (``int_algorithm``), and are differentiable with
    autograd to any order. The overlap is evaluated by the driver itself
    (:meth:`eval_matrix`); the dipole and quadrupole integrals by
    :class:`~dxtb._src.integral.driver.pytorch.DipolePytorch` and
    :class:`~dxtb._src.integral.driver.pytorch.QuadrupolePytorch`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._algorithm = DEFAULT_ALGORITHM
        self._positions: Tensor
        self._positions_single: Tensor
        self._positions_batch: list[Tensor]
        self._basis_batch: list[Basis]
        self._ihelp_batch: list[IndexHelper]
        # structural plans of the pair builder, owned by (and freed with) the
        # driver; the single-molecule plan is tied to the helper it was built for
        self._plan_single: tuple[IndexHelper, PairPlan] | None = None
        self._plan_batch: list[PairPlan] = []

    def setup(self, positions: Tensor, **kwargs: Any) -> None:
        """
        Run the `libcint`-specific driver setup.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        """
        if self.ihelp.batch_mode == 0:
            # setup `Basis` class if not already done
            if self._basis is None:
                self.basis = Basis(
                    torch.unique(self.numbers),
                    self.par,
                    self.ihelp,
                    device=self.device,
                    dtype=self.dtype,
                )

            self._positions_single = positions
        else:
            mask = kwargs.get("mask")

            self._positions_batch = []
            self._basis_batch = []
            self._ihelp_batch = []
            self._plan_batch = []
            for _batch in range(self.numbers.shape[0]):
                # POSITIONS
                if self.ihelp.batch_mode == 1:
                    nums = deflate(self.numbers[_batch])

                    if mask is not None:
                        pos = torch.masked_select(
                            positions[_batch],
                            mask[_batch],
                        ).reshape((-1, 3))
                    else:
                        # padding is identified from the atomic numbers,
                        # since zero coordinates are ambiguous (e.g., an atom
                        # at the origin)
                        pos = positions[_batch, : nums.shape[-1]]

                elif self.ihelp.batch_mode == 2:
                    pos = positions[_batch]
                    nums = self.numbers[_batch]

                else:
                    raise ValueError(
                        f"Unknown batch mode '{self.ihelp.batch_mode}'."
                    )

                self._positions_batch.append(pos)

                # INDEXHELPER
                # unfortunately, we need a new IndexHelper for each batch,
                # but this is much faster than `calc_overlap`
                ihelp = IndexHelper.from_numbers(nums, self.par)

                self._ihelp_batch.append(ihelp)
                self._plan_batch.append(prepare(ihelp, self.device))

                # BASIS
                bas = Basis(
                    torch.unique(nums),
                    self.par,
                    ihelp,
                    dtype=self.dtype,
                    device=self.device,
                )

                self._basis_batch.append(bas)

        # setting positions signals successful setup; save current positions to
        # catch new positions and run the required re-setup of the driver
        self._positions = positions.detach().clone()

    @property
    def algorithm(self) -> str:
        """
        Name of the 1D kernel (``int_algorithm``) of the pair builder.
        Defaults to ``constants.labels.INTALGORITHM_DEFAULT``.
        """
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: str) -> None:
        get_kernel(value)  # validates the name
        self._algorithm = value.casefold()

    @property
    def kernel(self) -> Kernel1D:
        """1D kernel of the pair builder, selected by :attr:`algorithm`."""
        return get_kernel(self.algorithm)

    def eval_matrix(
        self, components: tuple[tuple[int, int, int], ...] | None = None
    ) -> Tensor:
        """
        AO integral matrix of the current setup, for single molecules and
        (zero-padded) batches alike.

        Parameters
        ----------
        components : tuple[tuple[int, int, int], ...] | None, optional
            Per-axis multipole order of every output component. ``None``
            (default) is the overlap.

        Returns
        -------
        Tensor
            Integral of shape ``(ncomp, norb, norb)`` (single) or
            ``(nbatch, ncomp, norb, norb)`` (batched).
        """
        kernel = self.kernel

        def _one(
            ihelp: IndexHelper, bas: Basis, pos: Tensor, plan: PairPlan
        ) -> Tensor:
            alphas, coeffs = bas.create_cgtos()
            return assemble_matrix(
                kernel, ihelp, alphas, coeffs, pos, components, plan=plan
            )

        if self.ihelp.batch_mode == 0:
            if (
                self._plan_single is None
                or self._plan_single[0] is not self.ihelp
            ):
                self._plan_single = (
                    self.ihelp,
                    prepare(self.ihelp, self.device),
                )

            return _one(
                self.ihelp,
                self.basis,
                self._positions_single,
                self._plan_single[1],
            )

        return pack(
            [
                _one(ihelp, bas, pos, plan)
                for ihelp, bas, pos, plan in zip(
                    self._ihelp_batch,
                    self._basis_batch,
                    self._positions_batch,
                    self._plan_batch,
                )
            ]
        )
