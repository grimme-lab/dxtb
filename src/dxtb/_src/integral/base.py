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
Integrals: Base Classes
=======================

Base class for integral classes and their actual implementations.
"""

from __future__ import annotations

import torch
from tad_mctc.math import einsum

from dxtb._src.constants import defaults
from dxtb._src.typing import Literal, PathLike, Self, Tensor, TensorLike

from .abc import IntegralABC
from .driver import IntDriver
from .utils import snorm

__all__ = ["BaseIntegral"]


class BaseIntegral(IntegralABC, TensorLike):
    """
    Base class for integral implementations.

    All integral calculations are executed by its child classes.
    """

    _matrix: Tensor | None
    """Internal storage variable for the integral matrix."""

    _gradient: Tensor | None
    """Internal storage variable for the cartesian gradient."""

    _norm: Tensor | None
    """Internal storage variable for the overlap norm."""

    family: str | None
    """Family of the integral implementation (PyTorch or libcint)."""

    uplo: Literal["n", "u", "l"] = "l"
    """
    Whether the matrix of unique shell pairs should be create as a
    triangular matrix (`l`: lower, `u`: upper) or full matrix (`n`).
    Defaults to `l` (lower triangular matrix).
    """

    cutoff: Tensor | float | int | None = defaults.INTCUTOFF
    """
    Real-space cutoff for integral calculation in Bohr. Defaults to
    `constants.defaults.INTCUTOFF`.
    """

    __slots__ = ["_matrix", "_gradient", "_norm"]

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        uplo: Literal["n", "N", "u", "U", "l", "L"] = "l",
        cutoff: Tensor | float | int | None = defaults.INTCUTOFF,
        _matrix: Tensor | None = None,
        _gradient: Tensor | None = None,
        _norm: Tensor | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.label = self.__class__.__name__

        self.cutoff = cutoff

        if uplo not in ("n", "N", "u", "U", "l", "L"):
            raise ValueError(f"Unknown option for `uplo` chosen: '{uplo}'.")
        self.uplo = uplo.casefold()  # type: ignore

        self._norm = _norm
        self._matrix = _matrix
        self._gradient = _gradient

    def checks(self, driver: IntDriver) -> None:
        """
        Check if the driver is setup.

        Parameters
        ----------
        driver : IntDriver
            Integral driver for the calculation.
        """
        if not driver.is_setup():
            raise RuntimeWarning(
                "Integral driver not setup. Run `driver.setup(positions)` "
                "before passing the driver to the integral build."
            )

        if "pytorch" in self.label.casefold():
            # pylint: disable=import-outside-toplevel
            from .driver.pytorch.driver import (
                BaseIntDriverPytorch as _BaseIntDriver,
            )

        elif "libcint" in self.label.casefold():
            # pylint: disable=import-outside-toplevel
            from .driver.libcint.driver import (
                BaseIntDriverLibcint as _BaseIntDriver,
            )

        else:
            raise RuntimeError(
                f"Unknown integral implementation: '{self.label}'."
            )

        if not isinstance(driver, _BaseIntDriver):
            raise RuntimeError(
                f"Wrong integral driver selected for '{self.label}'."
            )

    def clear(self) -> None:
        """
        Clear the integral matrix and gradient.
        """
        self._matrix = None
        self._norm = None
        self._gradient = None

    @property
    def requires_grad(self) -> bool:
        """
        Check if any field of the integral class is requires gradient.

        Returns
        -------
        bool
            Flag for gradient requirement.
        """
        for field in (self._matrix, self._gradient, self._norm):
            if field is not None and field.requires_grad:
                return True

        return False

    def normalize(self, norm: Tensor | None = None) -> None:
        """
        Normalize the integral (changes ``self.matrix``).

        Parameters
        ----------
        norm : Tensor | None, optional
            Overlap norm to normalize the integral.
        """
        if norm is None:
            if self.norm is not None:
                norm = self.norm
            else:
                norm = snorm(self.matrix)

        if norm.ndim == 1:
            einsum_str = "...ij,i,j->...ij"
        elif norm.ndim == 2:
            einsum_str = "b...ij,bi,bj->b...ij"
        else:
            raise ValueError(f"Invalid norm shape: {norm.shape}")

        self.matrix = einsum(einsum_str, self.matrix, norm, norm)

    def normalize_gradient(self, norm: Tensor | None = None) -> None:
        """
        Normalize the gradient (changes ``self.gradient``).

        Parameters
        ----------
        norm : Tensor
            Overlap norm to normalize the integral.
        """
        if norm is None:
            if self.norm is not None:
                norm = self.norm
            else:
                norm = snorm(self.matrix)

        einsum_str = "...ijx,...i,...j->...ijx"
        self.gradient = einsum(einsum_str, self.gradient, norm, norm)

    def to_pt(self, path: PathLike | None = None) -> None:
        """
        Save the integral matrix to a file.

        Parameters
        ----------
        path : PathLike | None
            Path to the file where the integral matrix should be saved. If
            ``None``, the matrix is saved to the default location.
        """
        if path is None:
            path = f"{self.label.casefold()}.pt"

        torch.save(self.matrix, path)

    def to(self, device: torch.device | None) -> Self:
        """
        Returns a copy of the integral on the specified device "``device``".

        This is essentially a wrapper around the :meth:`to` method of the
        :class:`TensorLike` class, but explicitly also moves the integral
        matrix.

        Parameters
        ----------
        device : torch.device | None
            Device to which all associated tensors should be moved.

        Returns
        -------
        Self
            A copy of the integral placed on the specified device.
        """
        if self._gradient is not None:
            self._gradient = self._gradient.to(device=device)

        if self._norm is not None:
            self._norm = self._norm.to(device=device)

        if self._matrix is not None:
            self._matrix = self._matrix.to(device=device)

        return super().to(device=device)

    @property
    def matrix(self) -> Tensor:
        if self._matrix is None:
            raise RuntimeError(
                "Integral matrix not found. This can be caused by two "
                "reasons:\n"
                "1. The integral has not been calculated yet.\n"
                "2. The integral was cleared, despite being required "
                "in a subsequent calculation. Check the cache settings."
            )
        return self._matrix

    @matrix.setter
    def matrix(self, mat: Tensor) -> None:
        self._matrix = mat

    @property
    def norm(self) -> Tensor | None:
        return self._norm

    @norm.setter
    def norm(self, n: Tensor) -> None:
        self._norm = n

    @property
    def gradient(self) -> Tensor:
        if self._gradient is None:
            raise RuntimeError("Integral gradient has not been calculated.")
        return self._gradient

    @gradient.setter
    def gradient(self, mat: Tensor) -> None:
        self._gradient = mat

    def __str__(self) -> str:  # pragma: no cover
        d = self.__dict__.copy()
        if self._matrix is not None:
            d["_matrix"] = self._matrix.shape
        if self._gradient is not None:
            d["_gradient"] = self._gradient.shape
        if self._norm is not None:
            d["_norm"] = self._norm.shape

        return f"{self.__class__.__name__}({d})"

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)
