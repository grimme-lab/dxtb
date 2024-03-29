"""
Definition of the repulsion contribution.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class EffectiveRepulsion(BaseModel):
    """
    Representation of the repulsion contribution for a parametrization.
    """

    kexp: float
    """
    Scaling of the interatomic distance in the exponential damping function of
    the repulsion energy.
    """

    klight: Optional[float] = None
    """
    Scaling of the interatomic distance in the exponential damping function of
    the repulsion energy for light elements, i.e., H and He (only GFN2).
    """


class Repulsion(BaseModel):
    """
    Possible repulsion parametrizations. Currently only the GFN1-xTB effective
    repulsion is supported.
    """

    effective: EffectiveRepulsion
    """Name of the represented method"""
