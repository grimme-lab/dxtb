# -*- coding: utf-8 -*-
# NOTE: THANKS TO THE COURTESY OF https://github.com/tbmalt/tbmalt
"""Unit conversion factors.

This module provides the factors for converting between different units of
measurement. These are the factors by which a value should be multiplied to
convert it into its atomic-unit equivalent. Conversion factors have been
grouped together by physical quantity, i.e. energy, length, dipole, etc.

Attributes:
    energy_units (Dict[str, int]): Energy unit conversion factors.
    length_units (Dict[str, int]): Length unit conversion factors.
    dipole_units (Dict[str, int]): Dipole unit conversion factors.

Notes:
    Available conversion factors for each quantity are provided below with
    their associated keys, all of which are in lower-case.

    Energy:
        - rydberg: 'rydberg' or 'ry'
        - electron-volt: "electronvolt" or 'ev'
        - kilocalorie per mol: 'kcal/mol'
        - kelvin: 'kelvin' or 'k'
        - reciprocal centimetre: 'cm^-1'
        - joule: 'joule' or 'j'
        - hartree / atomic-unit: 'hartree', 'ha' or 'au'

    Length:
        - angstrom: 'angstrom', 'aa' or 'a'
        - meter: 'meter' or 'm'
        - picometer: 'picometer' or 'pm'
        - bohr / atomic-unit: 'bohr', 'au'

    Dipole:
        - coulomb-meter: 'coulombmeter' or 'cm'
        - debye: 'debye' or 'd'
        - e-bohr / atomic-unit: 'ebohr', 'eb' or 'au'

    Conversion factors & physical constants are taken directly from DFTB+.

Examples:
    To convert from angstrom to atomic units (bohr):

    >>> from tbmalt.data.units import length_units
    >>> length_in_angstrom = 10.0
    >>> length_in_atomic_units = length_in_angstrom * length_units['angstrom']
    >>> print(length_in_angstrom)
    10.0
    >>> print(length_in_atomic_units)
    18.897259885789232

"""
from typing import Dict

# Developers Notes: The constants used here will eventually be abstracted to a
# separate module. Until then, the warning C0103 has been disabled to stop
# PyLint from complaining.

# Physical constants taken directly from DFTB+'s constants.F90 file. These
# are used to build all necessary conversion factors.
# pylint: disable=C0103
# Bohr --> Angstrom
_Bohr__AA = 0.529177249
# Angstrom --> Bohr
_AA__Bohr = 1.0 / _Bohr__AA
# Hartree --> eV
_Hartree__eV = 27.2113845
# eV --> Hartree
_eV__Hartree = 1.0 / _Hartree__eV
# Hartree --> Joule
_Hartree__J = 4.3597441775e-18
# Joule --> Hartree
_J__Hartree = 1.0 / _Hartree__J
# kcal/mol --> Hartree
_kcal_mol__Hartree = 0.0015946683859874898
# Rydberg --> m-1 codata 2006 R_infty
_Rydberg__m = 10973731.568527
# Hartree --> cm-1
_Hartree__cm = 2.0 * _Rydberg__m / 100.0
# K --> Hartree
_Boltzmann = 0.00000316681534524639
# Atomic units --> Coulomb
_au__Coulomb = 1.60217653e-19
# Coulomb --> Atomic units [electric charge]
_Coulomb__au = 1.0 / _au__Coulomb
# Fine structure constant
_alpha_fs = 0.007297352568
# Atomic units [time] --> femtoseconds
_au__fs = 0.02418884326505
# Debye -> Atomic units [dipole]
_Debye__au = 1.0e-16 * _alpha_fs * _au__fs * _Coulomb__au * _AA__Bohr**2

# Conversion factors for energy units
energy_units: Dict[str, float] = {
    "rydberg": 0.5,
    "ry": 0.5,
    "electronvolt": _eV__Hartree,
    "ev": _eV__Hartree,
    "kcal/mol": _kcal_mol__Hartree,
    "kelvin": _Boltzmann,
    "k": _Boltzmann,
    "cm^-1": 1.0 / _Hartree__cm,
    "joule": _J__Hartree,
    "j": _J__Hartree,
    "hartree": 1.0,
    "ha": 1.0,
    "au": 1.0,
}

# Conversion factors for length units
length_units: Dict[str, float] = {
    "angstrom": _AA__Bohr,
    "aa": _AA__Bohr,
    "a": _AA__Bohr,
    "meter": 1.0e10 * _AA__Bohr,
    "m": 1.0e10 * _AA__Bohr,
    "picometer": 1.0e-2 * _AA__Bohr,
    "pm": 1.0e-2 * _AA__Bohr,
    "bohr": 1.0,
    "au": 1.0,
}

# Conversion factors for dipole units
dipole_units: Dict[str, float] = {
    "cm": _Coulomb__au * 1.0e10 * _AA__Bohr,
    "coulombmeter": _Coulomb__au * 1.0e10 * _AA__Bohr,
    "debye": _Debye__au,
    "d": _Debye__au,
    "ebohr": 1.0,
    "eb": 1.0,
    "au": 1.0,
}
