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

Base class for Integrals classes and their actual implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.param import Param
from dxtb._src.typing import PathLike, Tensor, TensorLike

__all__ = [
    "BaseIntegralImplementation",
    "IntDriver",
    "IntegralContainer",
]


class IntDriver(TensorLike):
    """Base class for the integral driver."""

    numbers: Tensor
    """Atomic numbers of the atoms in the system."""

    par: Param
    """Representation of parametrization of xtb model."""

    ihelp: IndexHelper
    """Helper class for indexing."""

    __label: str
    """Identifier label for integral driver."""

    __slots__ = [
        "numbers",
        "par",
        "ihelp",
        "_basis",
        "_positions",
        "__label",
    ]

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        ihelp: IndexHelper,
        *,
        _basis: Basis | None = None,
        _positions: Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)
        self.numbers = numbers
        self.par = par
        self.ihelp = ihelp
        self._basis = _basis
        self._positions = _positions
        self.__label = self.__class__.__name__

    @property
    def label(self) -> str:
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
                "`driver.invalidate()`, `integrals.reset_all()` or \n"
                "`integrals.invalidate_driver()`, or `calc.reset_all()` \n"
                "after the first autograd run.\n"
            ) from e

        tol = torch.finfo(positions.dtype).eps ** 0.75 if tol is None else tol
        if diff.abs().sum() > tol:
            return False

        return True

    def invalidate(self) -> None:
        """
        Invalidate the integral driver to require new setup.
        """
        self._positions = None

    def is_setup(self) -> bool:
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

    def __str__(self) -> str:
        dict_repr = []
        for key, value in self.__dict__.items():
            if isinstance(value, Tensor):
                value_repr = f"{value.shape}"
            else:
                value_repr = repr(value)
            dict_repr.append(f"    {key}: {value_repr}")
        dict_str = "{\n" + ",\n".join(dict_repr) + "\n}"
        return f"{self.__class__.__name__}({dict_str})"

    def __repr__(self) -> str:
        return str(self)


#########################################################


class IntegralImplementationABC(ABC):
    """
    Abstract base class for (actual) integral implementations.

    All integral calculations are executed by this class.
    """

    @abstractmethod
    def build(self, driver: IntDriver) -> Tensor:
        """
        Create the integral matrix.

        Parameters
        ----------
        driver : IntDriver
            Integral driver for the calculation.

        Returns
        -------
        Tensor
            Integral matrix.
        """

    @abstractmethod
    def get_gradient(self, driver: IntDriver) -> Tensor:
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


class BaseIntegralImplementation(IntegralImplementationABC, TensorLike):
    """
    Base class for (actual) integral implementations.

    All integral calculations are executed by this class.
    """

    __slots__ = ["_matrix", "_norm", "_gradient"]

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        normalize: bool = True,
        _matrix: Tensor | None = None,
        _norm: Tensor | None = None,
        _gradient: Tensor | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.label = self.__class__.__name__

        self.normalize = normalize
        self._matrix = _matrix
        self._norm = _norm
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

    @property
    def matrix(self) -> Tensor:
        if self._matrix is None:
            raise RuntimeError("Integral matrix has not been calculated.")
        return self._matrix

    @matrix.setter
    def matrix(self, mat: Tensor) -> None:
        self._matrix = mat

    @property
    def norm(self) -> Tensor:
        if self._norm is None:
            raise RuntimeError("Overlap norm has not been calculated.")
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

    def __str__(self) -> str:
        d = self.__dict__.copy()
        if self._matrix is not None:
            d["_matrix"] = self._matrix.shape
        if self._norm is not None:
            d["_norm"] = self._norm.shape
        if self._gradient is not None:
            d["_gradient"] = self._gradient.shape

        return f"{self.__class__.__name__}({d})"

    def __repr__(self) -> str:
        return str(self)


#########################################################


class IntegralContainer(TensorLike):
    """
    Base class for integral container.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        _run_checks: bool = True,
    ):
        super().__init__(device, dtype)
        self._run_checks = _run_checks

    @property
    def run_checks(self) -> bool:
        return self._run_checks

    @run_checks.setter
    def run_checks(self, run_checks: bool) -> None:
        current = self.run_checks
        self._run_checks = run_checks

        # switching from False to True should automatically run checks
        if current is False and run_checks is True:
            self.checks()

    @abstractmethod
    def checks(self) -> None:
        """Run checks for integrals."""
