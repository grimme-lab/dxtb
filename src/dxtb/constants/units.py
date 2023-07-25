"""
Unit conversion factors.
"""

from scipy.constants import physical_constants


def get_constant(constant_name: str) -> float:
    if constant_name not in physical_constants:
        raise KeyError(f"Constant '{constant_name}' not found.")
    return physical_constants[constant_name][0]


class CODATA:
    """CODATA values for various physical constants"""

    h = get_constant("Planck constant")
    """Planck's constant"""

    c = get_constant("speed of light in vacuum")
    """Speed of light in vacuum (m/s)"""

    kb = get_constant("Boltzmann constant")
    """Boltzmann's constant"""

    na = get_constant("Avogadro constant")
    """Avogadro's number (mol^-1)"""

    e = get_constant("elementary charge")
    """Elementary charge"""

    alpha = get_constant("fine-structure constant")
    """Fine structure constant (CODATA2018)"""

    me = get_constant("electron mass")
    """Rest mass of the electron (kg)"""

    bohr = get_constant("Bohr radius")
    """Bohr radius (m)"""


PI = 3.1415926535897932384626433832795029

# LENGTH

AU2METER = CODATA.bohr
"""
Conversion from bohr (a.u.) to meter.
This equals: 1 bohr = 5.29177210903e-11 m.
"""

METER2AU = 1.0 / AU2METER
"""Conversion from meter to atomic units."""

AA2METER = 1e-10
"""Factor for conversion from Angstrom to meter (1e-10)."""

AA2AU = AA2METER * METER2AU
"""
Factor for conversion from angstrom to atomic units.
This equals: 1 Angstrom = 1.8897261246204404 a.u.
"""

# ENERGY

EV2AU = 1.0 / 27.21138505
"""Factor for conversion from eletronvolt to atomic units."""

K2AU = 3.166808578545117e-06
"""Factor for conversion from Kelvin to atomic units for electronic temperatur."""

AU2KCAL = 627.5096080305927
"""Factor for conversion from atomic units (Hartree) to kcal/mol."""

CAL2J = 4.184
"""Factor for conversion from Calorie to Joule"""

AU2COULOMB = CODATA.e
"""
Factor for conversion from atomic units to Coulomb.
This equals the elementary charge.
"""

COULOMB2AU = 1.0 / AU2COULOMB
"""Factor for conversion from Coulomb to atomic units."""

AU2JOULE = get_constant("atomic unit of energy")
"""Factor for conversion from atomic units to Joule (J = kg·m²·s⁻²)."""

JOULE2AU = 1.0 / AU2JOULE
"""
Factor for conversion from Joule (J = kg·m²·s⁻²) to atomic units.
This equals: 1 Joule = 2.294e+17 Hartree.
Could also be calculated as: CODATA.me * CODATA.c**2 * CODATA.alpha**2
"""

VAA2AU = JOULE2AU / (COULOMB2AU * AA2AU)
"""Factor for conversion from  V/Å = J/(C·Å) to atomic units"""


AU_KG = CODATA.me
KG_AU = 1.0 / AU_KG

AMU_KG = get_constant("unified atomic mass unit")
KG_AMU = 1.0 / AMU_KG

AMU_AU = AMU_KG * KG_AU
AU_AMU = AU_KG * KG_AU


AU2GMOL = 1e3 * CODATA.me * CODATA.na
"""Electron mass (a.u.) to molecular mass per mole (g/mol)"""

GMOL2AU = 1.0 / AU2GMOL
"""Molecular mass per mole (g/mol) to electron mass (a.u.)"""


AU2RCM = AU2JOULE / (CODATA.h * CODATA.c) * 1e-2
"""Conversion from hartree to reciprocal centimeters"""

RCM2AU = 1.0 / AU2RCM
"""Conversion from reciprocal centimeters to hartree"""


DEBYE2AU = 1e-21 / CODATA.c * COULOMB2AU * METER2AU
"""
Conversion from Debye to atomic units.
This equals: 1 Debye in SI units (C m) = 0.393430 a.u.
"""

AU2DEBYE = 1.0 / DEBYE2AU
"""Conversion from atomic units to Debye."""

AMU = 5.485799090649e-4
AU2KMMOL = (DEBYE2AU / AA2AU) ** 2 / AMU * 42.256

# TIME

AU2SECOND = get_constant("atomic unit of time")
"""Conversion from atomic units to second. Atomic unit of time (s)."""

SECOND2AU = 1.0 / AU2SECOND
"""Conversion from second to atomic units."""
