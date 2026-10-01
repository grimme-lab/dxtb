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

from abc import abstractmethod

import torch

from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.typing import Any, Literal, Tensor

from ...base import IntDriver
from .base import PytorchImplementation
from .impls.kernels import DEFAULT_ALGORITHM, get_kernel
from .impls.pairs import assemble_matrix, assemble_overlap_gradient
from .impls.pipeline import Kernel1D

__all__ = [
    "BaseIntDriverPytorch",
    "IntDriverPytorch",
    "IntDriverPytorchLegacy",
]


class BaseIntDriverPytorch(PytorchImplementation, IntDriver):
    """
    PyTorch-based integral driver.

    Note
    ----
    The overlap and its gradient are evaluated by the driver itself
    (:meth:`eval_ovlp`, :meth:`eval_ovlp_grad`); the dipole and quadrupole
    integrals are built by
    :class:`~dxtb._src.integral.driver.pytorch.DipolePytorch` and
    :class:`~dxtb._src.integral.driver.pytorch.QuadrupolePytorch` with the
    pair builder and the kernel selected in :attr:`algorithm`
    (``int_algorithm``).
    """

    algorithm: str | None = None
    """
    Name of the 1D kernel (``int_algorithm``) of the pair builder.
    ``None``: the default kernel (``os``).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._positions: Tensor
        self._positions_single: Tensor
        self._positions_batch: list[Tensor]
        self._basis_batch: list[Basis]
        self._ihelp_batch: list[IndexHelper]

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

            self._positions_batch = []
            self._basis_batch = []
            self._ihelp_batch = []
            for _batch in range(self.numbers.shape[0]):
                # POSITIONS
                if self.ihelp.batch_mode == 1:
                    # pylint: disable=import-outside-toplevel
                    from tad_mctc.batch import deflate

                    nums = deflate(self.numbers[_batch])

                    mask = kwargs.pop("mask", None)
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
    def kernel(self) -> Kernel1D:
        """1D kernel of the pair builder, selected by :attr:`algorithm`."""
        return get_kernel(self.algorithm or DEFAULT_ALGORITHM)

    @abstractmethod
    def eval_ovlp(
        self,
        positions: Tensor,
        bas: Basis,
        ihelp: IndexHelper,
        uplo: Literal["n", "u", "l"] = "l",
        cutoff: Tensor | float | int | None = None,
    ) -> Tensor:
        """
        Overlap of one molecule.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(nat, 3)``).
        bas : Basis
            Basis set information.
        ihelp : IndexHelper
            Helper class for indexing.
        uplo : Literal["n", "u", "l"], optional
            Which triangle of the matrix is computed and mirrored.
        cutoff : Tensor | float | int | None, optional
            Real-space cutoff for the integral calculation in Bohr.

        Returns
        -------
        Tensor
            Overlap matrix of shape ``(norb, norb)``.
        """

    @abstractmethod
    def eval_ovlp_grad(
        self,
        positions: Tensor,
        bas: Basis,
        ihelp: IndexHelper,
        uplo: Literal["n", "u", "l"] = "l",
        cutoff: Tensor | float | int | None = None,
    ) -> Tensor:
        """
        Overlap gradient of one molecule (same arguments as
        :meth:`eval_ovlp`).

        Returns
        -------
        Tensor
            Derivative of every overlap element :math:`S_{ij}` with respect
            to the position of the atom of orbital :math:`i`, shape
            ``(norb, norb, 3)``.
        """


class IntDriverPytorch(BaseIntDriverPytorch):
    """
    PyTorch-based integral driver.

    All integrals are built by the pair builder with the kernel selected in
    :attr:`algorithm`, and are differentiable with autograd to any order.
    The overlap gradient is computed analytically from the same kernel.
    ``uplo`` and ``cutoff`` have no effect: the full matrix is always built.
    """

    def eval_ovlp(
        self,
        positions: Tensor,
        bas: Basis,
        ihelp: IndexHelper,
        uplo: Literal["n", "u", "l"] = "l",
        cutoff: Tensor | float | int | None = None,
    ) -> Tensor:
        alphas, coeffs = bas.create_cgtos()
        return assemble_matrix(self.kernel, ihelp, alphas, coeffs, positions)[0]

    def eval_ovlp_grad(
        self,
        positions: Tensor,
        bas: Basis,
        ihelp: IndexHelper,
        uplo: Literal["n", "u", "l"] = "l",
        cutoff: Tensor | float | int | None = None,
    ) -> Tensor:
        alphas, coeffs = bas.create_cgtos()
        return assemble_overlap_gradient(
            self.kernel, ihelp, alphas, coeffs, positions
        )


class IntDriverPytorchLegacy(BaseIntDriverPytorch):
    """
    PyTorch-based integral driver with the old loop-based version of the
    overlap matrix build, using the explicit McMurchie-Davidson E-coefficients
    for every shell pair. It has no overlap gradient. The multipole integrals
    are built by the pair builder, as for :class:`IntDriverPytorch`.
    """

    def eval_ovlp(
        self,
        positions: Tensor,
        bas: Basis,
        ihelp: IndexHelper,
        uplo: Literal["n", "u", "l"] = "l",
        cutoff: Tensor | float | int | None = None,
    ) -> Tensor:
        # pylint: disable=import-outside-toplevel
        from .impls.legacy import overlap_legacy

        return overlap_legacy(positions, bas, ihelp, uplo, cutoff)

    def eval_ovlp_grad(
        self,
        positions: Tensor,
        bas: Basis,
        ihelp: IndexHelper,
        uplo: Literal["n", "u", "l"] = "l",
        cutoff: Tensor | float | int | None = None,
    ) -> Tensor:
        # pylint: disable=import-outside-toplevel
        from .impls.legacy import overlap_gradient_legacy

        return overlap_gradient_legacy(positions, bas, ihelp, uplo, cutoff)
