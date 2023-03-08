# This file is part of xtbml.
"""
Definition of the dispersion contribution.
"""
from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel

from .._types import Tensor
from ..constants import xtb


class D3Model(BaseModel):

    """
    Representation of the DFT-D3(BJ) contribution for a parametrization.
    """

    class Config:
        arbitrary_types_allowed = True

    s6: Union[float, Tensor] = xtb.DEFAULT_DISP_S6
    """Scaling factor for dipole-dipole term."""

    s8: Union[float, Tensor] = xtb.DEFAULT_DISP_S8
    """Scaling factor for dipole-quadrupole term."""

    s9: Union[float, Tensor] = xtb.DEFAULT_DISP_S9
    """Scaling factor for the many-body dispersion term (ATM/RPA-like)."""

    s10: Union[float, Tensor] = xtb.DEFAULT_DISP_S10
    """Scaling factor for quadrupole-quadrupole term."""

    a1: Union[float, Tensor]
    """Becke-Johnson damping parameter."""

    a2: Union[float, Tensor]
    """Becke-Johnson damping parameter."""


class Dispersion(BaseModel):
    """
    Possible dispersion parametrizations. Currently only the DFT-D3(BJ) is
    supported.
    """

    d3: Optional[D3Model] = None
    """Name of the represented method."""

    d4: Optional[D3Model] = None
    """D4 model for dispersion (not implemented)."""
