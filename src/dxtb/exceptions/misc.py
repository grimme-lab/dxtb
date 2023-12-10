"""
Custom warnings.
"""

__all__ = ["ParameterWarning", "ToleranceWarning"]


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
