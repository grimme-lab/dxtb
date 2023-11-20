"""
Base class for Integrals.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .._types import Any, Tensor, TensorLike
from .driver import IntDriver


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
            Cartesian coordinates of all atoms in the system (nat, 3).

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

    integral: BaseIntegralImplementation
    """Instance of actual integral type."""

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
        `build` method of the underlying `BaseIntegralType`.

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

    @property
    def matrix(self) -> Tensor:
        """
        Shortcut for matrix representation of the integral.

        Returns
        -------
        Tensor
            Integral matrix.

        Raises
        ------
        RuntimeError
            Integral has not been calculated yet.
        """
        if self.integral._matrix is None:
            raise RuntimeError("Integral matrix has not been calculated.")
        return self.matrix

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


class BaseIntegralImplementation(IntegralImplementationABC, TensorLike):
    """
    Base class for (actual) integral implementations.

    All integral calculations are executed by this class.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.label = self.__class__.__name__

        self._matrix = None
        self._norm = None

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
