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
    def __init__(self):
        self.message = "[Fatal] Moments higher than g are not supported"
        super().__init__(self.message)


class CgtoNegativeExponentsError(ValueError):
    def __init__(self):
        self.message = "Negative exponents not allowed"
        super().__init__(self.message)


class CgtoQuantumNumberError(ValueError):
    def __init__(self):
        self.message = "Only QN up to 6 supported"
        super().__init__(self.message)


class CgtoAzimudalQuantumNumberError(ValueError):
    def __init__(self):
        self.message = "No QM h-functions available"
        super().__init__(self.message)


class CgtoMaxPrimitivesError(ValueError):
    def __init__(self):
        self.message = "Max number of primitives is 6"
        super().__init__(self.message)
