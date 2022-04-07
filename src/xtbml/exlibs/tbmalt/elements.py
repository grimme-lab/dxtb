# -*- coding: utf-8 -*-
# NOTE: THANKS TO THE COURTESY OF https://github.com/tbmalt/tbmalt
"""Elemental reference data.

Reference data pertaining to chemical elements & their properties are located
here. As the `chemical_symbols` & `atomic_numbers` attributes are frequently
used they have been made accessible from the `tbmalt.data` namespace.

Attributes:
    chemical_symbols (List[str]): List of chemical symbols whose indices are
        the atomic numbers of the associated elements; i.e.
        `chemical_symbols[6]` will yield `"C"`.
    atomic_numbers (Dict[str, int]): Dictionary keyed by chemical symbols &
        valued by atomic numbers. This is used to get the atomic number
        associated with a given chemical symbol.

"""
from typing import List, Dict

# Chemical symbols of the elements. Neutronium is included to ensure the index
# matches the atomic number and to assist with batching behaviour.
chemical_symbols: List[str] = [
    # Period zero
    "X",
    # Period one
    "H",
    "He",
    # Period two
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    # Period three
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    # Period four
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    # Period five
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    # Period six
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    # Period seven
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

# Dictionary for looking up an element's atomic number.
atomic_numbers: Dict[str, int] = {sym: z for z, sym in enumerate(chemical_symbols)}
