"""
Unit conversion factors.
"""

AA2AU = 1.8897261246204404
"""Factor for conversion from angstrom to atomic units."""

EV2AU = 1.0 / 27.21138505
"""Factor for conversion from eletronvolt to atomic units."""

K2AU = 3.166808578545117e-06
"""Factor for conversion from Kelvin to atomic units for electronic temperatur."""

AU2KCAL = 627.5096080305927
"""Factor for conversion from atomic units (Hartree) to kcal/mol."""

CAL2J = 4.184
"""Factor for conversion from Calorie to Joule"""

C2AU = 1.0 / 1.60217653e-19
"""Factor for conversion from Coulomb to to atomic units"""

J2AU = 1.0 / 4.3597441775e-18
"""Factor for conversion from Joule to atomic units"""

VAA2AU = J2AU / (C2AU * AA2AU)
"""Factor for conversion from  V/Å = J/(C·Å) to atomic units"""
