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
    """Scaling factor for multipolar (dipole-dipole contribution) terms."""

    s8: Union[float, Tensor] = xtb.DEFAULT_DISP_S8
    """Scaling factor for multipolar (dipole-quadrupole contribution) terms."""

    s9: Union[float, Tensor] = xtb.DEFAULT_DISP_S9
    """Scaling factor for the many-body dispersion term (ATM/RPA-like)."""

    a1: Union[float, Tensor] = xtb.DEFAULT_DISP_A1
    """Becke-Johnson damping parameter."""

    a2: Union[float, Tensor] = xtb.DEFAULT_DISP_A2
    """Becke-Johnson damping parameter."""


class D4Model(BaseModel):
    """
    Representation of the DFT-D4 contribution for a parametrization.
    """

    class Config:
        arbitrary_types_allowed = True

    sc: bool = False
    """Whether the dispersion correctio is used self-consistently or not."""

    s6: Union[float, Tensor] = xtb.DEFAULT_DISP_S6
    """Scaling factor for multipolar (dipole-dipole contribution) terms"""

    s8: Union[float, Tensor] = xtb.DEFAULT_DISP_S8
    """Scaling factor for multipolar (dipole-quadrupole contribution) terms"""

    s9: Union[float, Tensor] = xtb.DEFAULT_DISP_S9
    """Scaling factor for the many-body dispersion term (ATM/RPA-like)."""

    s10: Union[float, Tensor] = xtb.DEFAULT_DISP_S10
    """Scaling factor for quadrupole-quadrupole term."""

    s10: Union[float, Tensor] = xtb.DEFAULT_DISP_S10
    """Scaling factor for quadrupole-quadrupole contribution."""

    alp: Union[float, Tensor] = xtb.DEFAULT_DISP_ALP
    """Exponent of zero damping function in the ATM term."""

    a1: Union[float, Tensor] = xtb.DEFAULT_DISP_A1
    """Becke-Johnson damping parameter."""

    a2: Union[float, Tensor] = xtb.DEFAULT_DISP_A2
    """Becke-Johnson damping parameter."""


class Dispersion(BaseModel):
    """
    Possible dispersion parametrizations. Currently, the DFT-D3(BJ) and DFT-D4
    methods are supported.
    """

    d3: Optional[D3Model] = None
    """D3 model for the dispersion."""

    d4: Optional[D4Model] = None
    """D4 model for the dispersion."""
