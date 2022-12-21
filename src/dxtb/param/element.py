# This file is part of xtbml.
"""
Element parametrization record containing the adjustable parameters for each species.
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel

list_str = List[str]
list_int = List[int]
list_float = List[float]


class Element(BaseModel):
    """
    Representation of the parameters for a species.
    """

    zeff: float
    """Effective nuclear charge used in repulsion"""
    arep: float
    """Repulsion exponent"""

    en: float
    """Electronnegativity"""

    shells: List[str]
    """Included shells with principal quantum number and angular momentum"""
    ngauss: List[int]
    """
    Number of primitive Gaussian functions used in the STO-NG expansion for
    each shell
    """
    levels: List[float]
    """Atomic level energies for each shell"""
    slater: List[float]
    """Slater exponents of the STO-NG functions for each shell"""
    refocc: List[float]
    """Reference occupation for each shell"""
    kcn: List[float]
    """CN dependent shift of the self energy for each shell"""
    shpoly: List[float]
    """Polynomial enhancement for Hamiltonian elements"""

    gam: float
    """Chemical hardness / Hubbard parameter"""
    lgam: List[float]
    """Relative chemical hardness for each shell"""
    gam3 = 0.0
    """Atomic Hubbard derivative"""

    dkernel: float = 0.0
    """Dipolar exchange-correlation kernel"""
    qkernel: float = 0.0
    """Quadrupolar exchange-correlation kernel"""
    mprad: float = 0.0
    """Multipole radius"""
    mpvcn: float = 0.0
    """Multipole valence CN"""

    xbond: float = 0.0
    """Halogen bonding strength"""


Elements = Dict[str, Element]
