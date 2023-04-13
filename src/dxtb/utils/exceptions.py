"""
Contains custom exceptions and warnings.
"""
from __future__ import annotations


class ParameterWarning(UserWarning):
    """
    Warning for when a parameter is not set.
    """


class TimerError(Exception):
    """
    A custom exception used to report errors in use of Timer class.
    """


class ToleranceWarning(UserWarning):
    """
    Warning for unreasonable tolerances.

    If tolerances are too small, the previous step in xitorch's Broyden method
    may become equal to the current step. This leads to a difference of zero,
    which in turn causes `NaN`s due to division by the difference.
    """


class IntegralTransformError(ValueError):
    def __init__(self) -> None:
        self.message = "[Fatal] Moments higher than g are not supported"
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
    def __init__(self, l: int) -> None:
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
