"""
Base class for Integrals.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .._types import Any, Tensor, TensorLike
from ..basis import Basis, IndexHelper
from ..param import Param


class IntDriver(TensorLike):
    """Base class for the integral driver."""

    numbers: Tensor
    """Atomic numbers of the atoms in the system."""

    par: Param
    """Representation of parametrization of xtb model."""

    ihelp: IndexHelper
    """Helper class for indexing."""

    label: str
    """Identifier label for integral driver."""

    family: int
    """Label for integral implementation family"""

    __slots__ = [
        "numbers",
        "par",
        "ihelp",
        "label",
        "_basis",
        "_positions",
    ]

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        ihelp: IndexHelper,
        *,
        basis: Basis | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)
        self.numbers = numbers
        self.par = par
        self.ihelp = ihelp
        self.label = self.__class__.__name__
        self._basis = basis
        self._positions: Tensor | None = None

    @property
    def basis(self) -> Basis:
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
            Cartesian coordinates of all atoms in the system (nat, 3).

        Returns
        -------
        bool
            Flag for set up status.
        """
        if self._positions is None:
            return False

        tol = torch.finfo(positions.dtype).eps ** 0.75 if tol is None else tol
        if (self._positions - positions).abs().sum() > tol:
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
        Run the `libcint`-specific driver setup.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
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

    label: str
    """Identifier label for integral type."""

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

    def get_gradient(self, driver: IntDriver) -> Tensor:
        """
        Calculation of the nuclear integral derivative (matrix). This method
        only calls the `get_gradient` method of the underlying
        `BaseIntegralType`.

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
            Integral matrix or `None` if not calculated yet.
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

    def __str__(self) -> str:
        mat = self.integral._matrix
        matinfo = mat.shape if mat is not None else None
        d = {**self.__dict__, "matrix": matinfo}
        d.pop("label")
        return f"{self.label}({d})"

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

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.label = self.__class__.__name__

        self._matrix = None
        self._norm = None
        self._gradient = None

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
    ):
        super().__init__(device, dtype)
        self._run_checks = True

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
