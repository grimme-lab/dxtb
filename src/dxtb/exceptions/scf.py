"""
Exceptions for the SCF procedure.
"""

__all__ = ["SCFConvergenceError", "SCFConvergenceWarning"]


class SCFConvergenceError(RuntimeError):
    def __init__(self, msg) -> None:
        self.message = msg
        super().__init__(self.message)


class SCFConvergenceWarning(RuntimeWarning):
    """
    Warning for failed SCF convergence.
    """
