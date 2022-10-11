"""
Contains custom exceptions and warnings.
"""


class ParameterWarning(UserWarning):
    """
    Warning for when a parameter is not set.
    """

class TimerError(Exception):
    """
    A custom exception used to report errors in use of Timer class.
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
