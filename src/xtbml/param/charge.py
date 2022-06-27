"""
Definition of the isotropic second-order charge interactions.
"""

from pydantic import BaseModel


class ChargeEffective(BaseModel):
    """
    Representation of the isotropic second-order charge interactions for a parametrization.
    """

    gexp: float = 2.0
    """Exponent of Coulomb kernel. """

    average: str = "harmonic"
    """Averaging function for Hubbard parameter."""


class Charge(BaseModel):
    """
    Possible charge parametrizations. Currently only the interaction kernel for the Klopman-Ohno electrostatics (effective) is supported.
    """

    effective: ChargeEffective
    """Klopman-Ohno electrostatics."""
