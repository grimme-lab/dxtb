"""
Contains custom exceptions and warnings.
"""
from __future__ import annotations

from .._types import Tensor


class DtypeError(ValueError):
    """
    Error for wrong data type of tensor.
    """


class DeviceError(RuntimeError):
    """
    Error for wrong device of tensor.
    """


class ParameterWarning(UserWarning):
    """
    Warning for when a parameter is not set.
    """


class ToleranceWarning(UserWarning):
    """
    Warning for unreasonable tolerances.

    If tolerances are too small, the previous step in xitorch's Broyden method
    may become equal to the current step. This leads to a difference of zero,
    which in turn causes `NaN`s due to division by the difference.
    """


class SCFConvergenceError(RuntimeError):
    def __init__(self, msg) -> None:
        self.message = msg
        super().__init__(self.message)


class SCFConvergenceWarning(RuntimeWarning):
    """
    Warning for failed SCF convergence.
    """


class IntegralTransformError(ValueError):
    def __init__(self) -> None:
        self.message = "[Fatal] Moments higher than f are not supported"
        super().__init__(self.message)


class CGTOSlaterExponentsError(ValueError):
    def __init__(self) -> None:
        self.message = "Negative Slater exponents not allowed."
        super().__init__(self.message)


class CGTOPrincipalQuantumNumberError(ValueError):
    def __init__(self, n: int) -> None:
        self.message = f"Maximum principal QN supported is {n}."
        super().__init__(self.message)


class CGTOAzimuthalQuantumNumberError(ValueError):
    def __init__(self, l: int | Tensor) -> None:
        s = ["s", "p", "d", "f", "g", "h"][l]
        self.message = f"Maximum azimuthal QN supported is {l} ({s}-orbitals)."
        super().__init__(self.message)


class CGTOQuantumNumberError(ValueError):
    def __init__(self) -> None:
        self.message = (
            "Azimuthal QN 'l' and principal QN 'n' must adhere to "
            "l ∊ [n-1, n-2, ..., 1, 0]."
        )
        super().__init__(self.message)


class CGTOPrimitivesError(ValueError):
    def __init__(self) -> None:
        self.message = "Number of primitives must be between 1 and 6."
        super().__init__(self.message)
