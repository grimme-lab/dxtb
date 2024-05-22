"""
Integral Types: Base
====================

Base class for Integrals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from tad_mctc.exceptions import DtypeError

from dxtb._src.typing import Any, PathLike, Tensor, TensorLike, override

if TYPE_CHECKING:
    from ..base import BaseIntegralImplementation, IntDriver
del TYPE_CHECKING

__all__ = ["BaseIntegral"]


class IntegralABC(ABC):
    """
    Abstract base class for integrals.

    This class works as a wrapper for the actual integral, which is stored in
    the `integral` attribute of this class.
    """

    @abstractmethod
    def build(self, positions: Tensor, **kwargs: Any) -> Tensor:
        """
        Create the integral matrix. This method only calls the `build` method
        of the underlying `BaseIntegralType`.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).

        Returns
        -------
        Tensor
            Integral matrix.

        Note
        ----
        The matrix is returned and also saved internally in the `mat` attribute.
        """


class BaseIntegral(IntegralABC, TensorLike):
    """
    Base class for integrals.

    This class works as a wrapper for the actual integral, which is stored in
    the `integral` attribute of this class.
    """

    label: str
    """Identifier label for integral type."""

    integral: BaseIntegralImplementation
    """Instance of actual integral type."""

    __slots__ = ["integral"]

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.label = self.__class__.__name__

    def build(self, driver: IntDriver) -> Tensor:
        """
        Calculation of the integral (matrix). This method only calls the
        :meth:`build` method of the underlying
        :class:`BaseIntegralImplementation`.

        Parameters
        ----------
        driver : IntDriver
            The integral driver for the calculation.

        Returns
        -------
        Tensor
            Integral matrix.
        """
        return self.integral.build(driver)

    def get_gradient(self, driver: IntDriver) -> Tensor:
        """
        Calculation of the nuclear integral derivative (matrix). This method
        only calls the :meth:`get_gradient` method of the underlying
        :class:`BaseIntegralImplementation`.

        Parameters
        ----------
        driver : IntDriver
            The integral driver for the calculation.

        Returns
        -------
        Tensor
            Nuclear integral derivative matrix.
        """
        return self.integral.get_gradient(driver)

    @property
    def matrix(self) -> Tensor | None:
        """
        Shortcut for matrix representation of the integral.

        Returns
        -------
        Tensor | None
            Integral matrix or ``None`` if not calculated yet.
        """
        return self.integral.matrix

    @matrix.setter
    def matrix(self, mat: Tensor) -> None:
        """
        Shortcut for matrix representation of the integral.

        Parameters
        ----------
        mat : Tensor
            Integral matrix.
        """
        self.integral.matrix = mat

    def to_pt(self, path: PathLike | None = None) -> None:
        """
        Save the integral matrix to a file.

        Parameters
        ----------
        path : PathLike | None
            Path to the file where the integral matrix should be saved. If
            ``None``, the matrix is saved to the default location.
        """
        self.integral.to_pt(path)

    def type(self, dtype: torch.dtype) -> BaseIntegral:
        """
        Returns a copy of the :class:`BaseIntegral` instance with specified
        floating point type.

        This method overwrites the usual approach because the
        :class:`BaseIntegral` class only contains the integral, which has to be
        moved.

        Parameters
        ----------
        dtype : torch.dtype
            Floating point type.

        Returns
        -------
        BaseIntegral
            A copy of the :class:`BaseIntegral` instance with the specified
            dtype.

        Raises
        ------
        RuntimeError
            If the ``__slots__`` attribute is not set in the class.
        DtypeError
            If the specified dtype is not allowed.
        """
        if self.dtype == dtype:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `type` method requires setting ``__slots__`` in the "
                f"'{self.__class__.__name__}' class."
            )

        if dtype not in self.allowed_dtypes:
            raise DtypeError(
                f"Only '{self.allowed_dtypes}' allowed (received '{dtype}')."
            )

        self.integral = self.integral.type(dtype)
        self.override_dtype(dtype)
        return self

    @override
    def to(self, device: torch.device) -> BaseIntegral:
        """
        Returns a copy of the :class:`.BaseIntegral` instance on the specified
        device.

        This method overwrites the usual approach because the
        :class:`.BaseIntegral` class only contains the integral, which has to be
        moved.

        Parameters
        ----------
        device : torch.device
            Device to which all associated tensors should be moved.

        Returns
        -------
        BaseIntegral
            A copy of the :class:`.BaseIntegral` instance placed on the
            specified device.

        Raises
        ------
        RuntimeError
            If the ``__slots__`` attribute is not set in the class.
        """
        if self.device == device:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `to` method requires setting ``__slots__`` in the "
                f"'{self.__class__.__name__}' class."
            )

        self.integral = self.integral.to(device)
        self.override_device(device)
        return self

    def __str__(self) -> str:
        mat = self.integral._matrix
        matinfo = mat.shape if mat is not None else None
        d = {**self.__dict__, "matrix": matinfo}
        d.pop("label")
        return f"{self.label}({d})"

    def __repr__(self) -> str:
        return str(self)
