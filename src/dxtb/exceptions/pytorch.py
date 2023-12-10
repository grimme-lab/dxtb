"""
Exceptions related to PyTorch.
"""


class DtypeError(ValueError):
    """
    Error for wrong data type of tensor.
    """


class DeviceError(RuntimeError):
    """
    Error for wrong device of tensor.
    """
