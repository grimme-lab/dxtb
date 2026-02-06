"""
Components: Spin Polarisation
=====================

Tight-binding components for spin polarisation.
"""

from dxtb._src.components.interactions.spin import (
    SpinPolarisation as SpinPolarisation,
)
from dxtb._src.components.interactions.spin import (
    new_spinpolarisation as new_spinpolarisation,
)

__all__ = ["SpinPolarisation", "new_spinpolarisation"]
