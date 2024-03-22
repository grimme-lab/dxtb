"""
Definition of the isotropic third-order onsite correction.
"""

from __future__ import annotations

from typing import Union

from pydantic import BaseModel


class ThirdOrderShell(BaseModel):
    """Representation of shell-resolved third-order electrostatics."""

    s: float
    """Scaling factor for s-orbitals."""

    p: float
    """Scaling factor for p-orbitals."""

    d: float
    """Scaling factor for d-orbitals."""


class ThirdOrder(BaseModel):
    """
    Representation of the isotropic third-order onsite correction.
    """

    shell: Union[bool, ThirdOrderShell] = False
    """Whether the third order contribution is shell-dependent or only atomwise."""
