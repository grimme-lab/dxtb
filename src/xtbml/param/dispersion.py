# This file is part of xtbml.

"""
Definition of the dispersion contribution.
"""

from typing import Optional
from pydantic import BaseModel


class D3Model(BaseModel):
    """
    Representation of the DFT-D3(BJ) contribution for a parametrization.
    """

    s6: Optional[float] = 1.0
    """Scaling factor for multipolar (dipole-dipole contribution) terms"""

    s8: float
    """Scaling factor for multipolar (dipole-quadrupole contribution) terms"""

    s9: Optional[float] = 0.0
    """Scaling factor for the many-body dispersion term (ATM/RPA-like)"""

    a1: float
    """Becke-Johnson damping parameter"""

    a2: float
    """Becke-Johnson damping parameter"""


class Dispersion(BaseModel):
    """
    Possible dispersion parametrizations. Currently only the DFT-D3(BJ) is supported.
    """

    d3: D3Model
    """Name of the represented method"""
