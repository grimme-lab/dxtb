# This file is part of xtbml.

"""
Definition of the halogen binding contribution.
"""

from pydantic import BaseModel


class ClassicalHalogen(BaseModel):
    """
    Representation of the classical geometry dependent halogen-bond (XB) correction for a parametrization.
    """

    damping: float
    """Damping factor of attractive contribution in Lennard-Jones-like potential"""

    rscale: float
    """Global scaling factor for covalent radii of AX bond"""


class Halogen(BaseModel):
    """
    Possible halogen parametrizations.
    """

    classical: ClassicalHalogen
    """Classical halogen-bond correction used in GFN1-xTB"""
