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

from abc import abstractmethod

import torch

from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import Literal, Tensor, TensorLike

__all__ = ["IntDriver"]


class IntDriver(TensorLike):
    """Base class for the integral driver."""

    numbers: Tensor
    """Atomic numbers of the atoms in the system."""

    par: ParamModule
    """Representation of parametrization of xTB method."""

    ihelp: IndexHelper
    """Helper class for indexing."""

    family: Literal["PyTorch", "libcint"]
    """Label for integral implementation family."""

    __label: str
    """Identifier label for integral driver."""

    __slots__ = [
        "numbers",
        "par",
        "ihelp",
        "family",
        "_basis",
        "_positions",
        "__label",
    ]

    def __init__(
        self,
        numbers: Tensor,
        par: Param | ParamModule,
        ihelp: IndexHelper,
        *,
        _basis: Basis | None = None,
        _positions: Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)
        self.numbers = numbers
        self.ihelp = ihelp
        self.par = (
            ParamModule(par, **self.dd) if isinstance(par, Param) else par
        )

        self._basis = _basis
        self._positions = _positions
        self.__label = self.__class__.__name__

    @property
    def label(self) -> str:
        """Label for the integral driver."""
        return self.__label

    @property
    def basis(self) -> Basis:
        """
        Get the basis class

        Returns
        -------
        Basis
            Basis class.

        Raises
        ------
        RuntimeError
            If the basis has not been set up.
        """
        if self._basis is None:
            raise RuntimeError("Basis has not been setup.")
        return self._basis

    @basis.setter
    def basis(self, bas: Basis) -> None:
        self._basis = bas

    def is_latest(self, positions: Tensor, tol: float | None = None) -> bool:
        """
        Check if the driver is set up and updated.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).

        Returns
        -------
        bool
            Flag for set up status.
        """
        if self._positions is None:
            return False

        try:
            diff = self._positions - positions
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                raise RuntimeError(
                    f"{e}\n\n"
                    "The positions tensor is on a different device than the \n"
                    "device on which the driver was setup. This usually \n"
                    "happens in CUDA runs with the CPU-enforced libcint \n"
                    "integral driver: The driver is setup on the CPU, but \n"
                    "the positions tensor supplied to the integral build is \n"
                    "on the GPU. Check the `build_<intgral>` or \n"
                    "`grad_<integral>` methods."
                ) from e

            raise RuntimeError(
                f"{e}\n\nThis is likely a functorch error that appears when \n"
                "running autograd twice without resetting certain cached \n"
                "values. It appears first in the integral driver. Depending \n"
                "on which level you interact with the API, use \n"
                "`driver.invalidate()`, `integrals.reset_all()`, \n"
                "`integrals.invalidate_driver()`, or `calc.reset_all()` \n"
                "after the first autograd run.\n"
            ) from e

        tol = torch.finfo(positions.dtype).eps ** 0.75 if tol is None else tol
        if diff.abs().sum() > tol:  # type: ignore[union-attr]
            return False

        return True

    def invalidate(self) -> None:
        """
        Invalidate the integral driver to require new setup.
        """
        self._positions = None

    def is_setup(self) -> bool:
        """Check if the driver is set up."""
        return self._positions is not None

    @abstractmethod
    def setup(self, positions: Tensor, **kwargs) -> None:
        """
        Run the specific driver setup.

        Example: For the `libcint` driver, the setup builds the basis in the
        format the `libcint` wrapper expects.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        """

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"Family: {self.family}, "
            f"Number of Atoms: {self.numbers.shape[-1]}, "
            f"Setup?: {self.is_setup()})"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)
