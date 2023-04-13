"""
Definition of the anisotropic second-order multipolar interactions.
"""
from __future__ import annotations

from typing import Union

from pydantic import BaseModel

from .._types import Tensor


class MultipoleDamped(BaseModel):
    """
    Representation of the anisotropic second-order multipolar interactions
    for a parametrization.
    """

    class Config:
        arbitrary_types_allowed = True

    dmp3: Union[float, Tensor]

    dmp5: Union[float, Tensor]

    kexp: Union[float, Tensor]

    shift: Union[float, Tensor]

    rmax: Union[float, Tensor]


class Multipole(BaseModel):
    """
    Possible parametrizations for multipole electrostatics.
    """

    damped: MultipoleDamped
    """Second-order multipolar electrostatics (GFN2)."""
